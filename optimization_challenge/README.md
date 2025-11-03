# Optimisation Challenge

## Objective
The objective of this task is to implement a optimisation based controller for the Franka Panda robot which should give joint velocities to help robot achieve a primary goal along with a secondary goal.

## Dependencies Installation
1. Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
2. Install [Moveit for ROS2 Humble](https://moveit.ai/install-moveit2/binary/)
3. Install [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html)
4. For optimisation solver you can either use Casadi in Pinocchio or [Nlopt](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/)
5. Franka ROS2 Integration [package](https://github.com/frankarobotics/franka_ros2)

## Task Instructions
Given a Franka Panda robot with an initial joint configuration, a target 6 dof endeffector goal and some secondary goal, implement a optimisation based controller which will give joint velocities to achieve the primary goal + secondary goal. For formulating the problem, the enviroment can be considered as obstacle free. Consider the endeffector tip for the franka to be "franka_hand".

- The primary goal of the robot is to reach a pose goal
- Seconday goal
	- Maintain a nominal set of joint angles.
	- Maximise the manipulability of the configuration while achieving primary goal

There are two secondary goals implement them one at a time, i.e one primary goal and one secondary out of the two mentioned.

Either of python or cpp can be used.

## Evaluation Criteria
- Residual errors in the primary goal,
- Motion Smoothness for joint velocity commands,
- Time for Convergence

## Submission
- The endeffector SE3() goal and the nominal set of joint angles should be given via a ROS2 action.
- Submit the assignment as ROS2 package which can be run by launching a launch file.