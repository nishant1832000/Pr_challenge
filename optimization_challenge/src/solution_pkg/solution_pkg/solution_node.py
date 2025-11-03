#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import signal
import sys
import pinocchio as pin
from casadi import SX, vertcat, Function, nlpsol, dot, det
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
from sensor_msgs.msg import JointState

class JointVelocityPublisher(Node):
    def __init__(self):
        super().__init__('joint_velocity_publisher')

        # Publisher for desired joint velocities
        self.publisher_ = self.create_publisher(Float64MultiArray, '/desired_joint_velocities', 10)
        self.sub_joint_state = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        package_name = "solution_pkg"
        pkg_path = get_package_share_directory(package_name)

        self.urdf_path = os.path.join(pkg_path, "urdf", "robot.urdf")
        
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        print(self.model)
        self.data = self.model.createData()

        self.frame_name = "fr3_link8" 
        self.frame_id = self.model.getFrameId(self.frame_name)

        self.q = np.zeros(self.model.nq)
        self.kp = 2.0
        self.lambda_m = 0.01
        self.x_des = np.array([0.4, 0.2, 0.5])
        self.R_des = pin.utils.rotate('z', np.pi/2)

        self.t = 0.0
        self.dt = 0.1
        self.num_joints = 7
        self.running = True

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.shutdown_handler)

        self.get_logger().info("JointVelocityPublisher started. Publishing to /desired_joint_velocities")

    def joint_state_callback(self, msg):
        # Use incoming joint state
        self.q = np.array(msg.position)
        

        # Compute desired joint velocities
        dq_sol = self.compute_optimal_dq(self.q)

        # Publish dq to velocity controller
        vel_msg = Float64MultiArray()
        vel_msg.data = dq_sol.tolist()
        self.publisher_.publish(vel_msg)


    def compute_optimal_dq(self, q_np):
        # Build CasADi symbolic problem
        q = SX.sym('q', self.model.nq)
        dq = SX.sym('dq', self.model.nv)
        p = SX.sym('p', 7)

        # Forward kinematics
        pin.forwardKinematics(self.model, self.data, q_np)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        x_ee = oMf.translation
        R_ee = oMf.rotation

        # Compute 6D Jacobian
        J6 = pin.computeFrameJacobian(self.model, self.data, q_np, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        

        # Orientation error (using log map)
        e_rot_vec = pin.log3(self.R_des.T @ R_ee)

        # Position error
        pos_error = self.x_des - x_ee
        e_total = vertcat(pos_error, e_rot_vec)

        # Desired task velocity
        xdot_des = self.kp * e_total
        task_error = J6 @ dq - xdot_des

        # Manipulability term
        J_lin = J6
        manip = SX.sqrt(det(J_lin @ J_lin.T) + 1e-6)

        print(manip)
        # Objective
        cost = dot(task_error, task_error) - self.lambda_m * manip

        print("cost :",cost)

        nlp = {'x': dq, 'p': p, 'f': cost}
        solver = nlpsol('solver', 'ipopt', nlp)

        # Solve optimization numerically
        sol = solver(x0=np.zeros(self.model.nv),p=q_np)

        print("sol :",sol)
        dq_sol = np.array(sol['x']).flatten()

        return dq_sol


   

    def publish_zero_velocity(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.num_joints
        self.publisher_.publish(msg)
        self.get_logger().info("Published zero velocities (safe stop).")

    def shutdown_handler(self, signum, frame):
        """Handle Ctrl+C — send zero velocities, then exit."""
        if self.running:
            self.running = False
            self.publish_zero_velocity()
            self.get_logger().info("Shutting down node after safe stop...")
            rclpy.shutdown()
            sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = JointVelocityPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_zero_velocity()
        node.get_logger().info("KeyboardInterrupt: sent zero velocities and shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
