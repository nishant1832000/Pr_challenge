"""
This module defines a `Planner` class for motion planning of a 3-DOF robotic arm in the presence of circular obstacles.

### Key Components:

- **ObstType**: A tuple representing a circular obstacle, defined as:
  - The obstacle's center as a tuple of (x, y) coordinates.
  - The obstacle's radius as a float.

- **JointState**: A tuple of three floats representing the robot's joint angles.

- **PlanningResult**: A tuple containing:
  - A boolean indicating whether a collision-free path was found.
  - A list of joint states representing the planned path from start to goal.

### Methods to implement:

- `collision_check(state, obstacles)`: Checks if a given joint state results in a collision with any obstacle.
- `plan(start, goal, obstacles)`: Computes a collision-free path from the start to the goal joint configuration, considering the obstacles.

Feel free to extend this class with helper functions to keep the code clean, efficient, and easy to maintain.
"""

import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ObstType = tuple[tuple[float, float], float]
JointState = tuple[float, float, float]
PlanningResult = tuple[bool, list[JointState]]


class Planner:
    def __init__(self, robot_link_lengths: list[float]):
        self.robot_link_lengths = robot_link_lengths
        self.joint_limits = [(-np.pi, np.pi) for _ in range(3)]  # Joint limits for 3 DOF
        
    def forward_kinematics(self, state: JointState) -> list:
        """
        Compute the forward kinematics for the 3-DOF arm.
        Returns the positions of all joints and end effector.
        """
        x, y = 0.0, 0.0  # Base position
        positions = [(x, y)]
        
        theta1, theta2, theta3 = state
        
        # First joint
        x += self.robot_link_lengths[0] * np.cos(theta1)
        y += self.robot_link_lengths[0] * np.sin(theta1)
        positions.append((x, y))
        
        # Second joint
        x += self.robot_link_lengths[1] * np.cos(theta1 + theta2)
        y += self.robot_link_lengths[1] * np.sin(theta1 + theta2)
        positions.append((x, y))
        
        # End effector (third joint)
        x += self.robot_link_lengths[2] * np.cos(theta1 + theta2 + theta3)
        y += self.robot_link_lengths[2] * np.sin(theta1 + theta2 + theta3)
        positions.append((x, y))
        
        return positions

    def collision_check(self, state: JointState, obstacles: list[ObstType]) -> bool:
        """
        Checks collision between a given state and obstacles

        :param      state:                The state
        :type       state:                JointState
        :param      obstacles:            The obstacles
        :type       obstacles:            list[ObstType]

        :returns:   True if collision, False otherwise
        :rtype:     bool
        """
        positions = self.forward_kinematics(state)
        
        # Check each segment of the robot arm for collision with obstacles
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            
            for obstacle in obstacles:
                center, radius = obstacle
                if self._segment_circle_collision(p1, p2, center, radius):
                    return True
                    
        return False

    def _segment_circle_collision(self, p1: tuple, p2: tuple, center: tuple, radius: float) -> bool:
        """
        Check if a line segment collides with a circle.
        """
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to circle center
        fx = center[0] - p1[0]
        fy = center[1] - p1[1]
        
        # Projection of f onto d
        dot_product = fx * dx + fy * dy
        segment_length_squared = dx * dx + dy * dy
        
        # If segment length is 0, just check point-circle collision
        if segment_length_squared == 0:
            return np.sqrt(fx*fx + fy*fy) <= radius
            
        # Find closest point on segment to circle center
        t = max(0, min(1, dot_product / segment_length_squared))
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        
        # Check if closest point is within circle
        distance = np.sqrt((closest_x - center[0])**2 + (closest_y - center[1])**2)
        return distance <= radius

    def _distance(self, state1: JointState, state2: JointState) -> float:
        """
        Calculate distance between two joint states.
        Uses weighted Euclidean distance in joint space.
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(state1, state2)))

    def _compute_dynamic_modulation_matrix(self, state: JointState, obstacles: list[ObstType],
                                         lambda0: float = 1.0, lambda1: float = 0.1,
                                         influence_distance: float = 1.0) -> np.ndarray:
        """
        Compute the Dynamic Modulation Matrix for obstacle avoidance.
        
        The DMM method modifies the velocity field by applying a modulation matrix
        that deforms the motion around obstacles while preserving stability.
        
        :param state: Current joint state
        :param obstacles: List of obstacles
        :param lambda0: Base eigenvalue for free motion
        :param lambda1: Eigenvalue for obstacle avoidance
        :param influence_distance: Distance at which obstacles start influencing motion
        :return: 3x3 modulation matrix M
        """
        # Start with identity matrix
        M = np.eye(3)
        
        positions = self.forward_kinematics(state)
        
        # For each arm segment and each obstacle, compute modulation
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            
            for obstacle in obstacles:
                center, radius = obstacle
                
                # Find closest point on segment to obstacle center
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                fx = center[0] - p1[0]
                fy = center[1] - p1[1]
                
                dot_product = fx * dx + fy * dy
                segment_length_squared = dx * dx + dy * dy
                
                if segment_length_squared > 0:
                    t = max(0, min(1, dot_product / segment_length_squared))
                    closest_x = p1[0] + t * dx
                    closest_y = p1[1] + t * dy
                    
                    # Vector from obstacle to closest point
                    obstacle_to_point = np.array([closest_x - center[0], closest_y - center[1]])
                    distance = np.linalg.norm(obstacle_to_point)
                    
                    # Check if obstacle is within influence distance
                    if distance < influence_distance + radius:
                        # Normalized direction vector
                        if distance > 1e-8:
                            e = obstacle_to_point / distance
                        else:
                            e = np.array([1.0, 0.0])  # Arbitrary direction if too close
                        
                        # Compute the Jacobian at the closest point
                        # We need to find which joint affects this point the most
                        jacobian = self._compute_point_jacobian(state, i, t)
                        
                        # Weight based on distance to obstacle
                        weight = max(0, 1 - (distance - radius) / influence_distance)
                        
                        # Outer product for modulation
                        if jacobian is not None and weight > 0:
                            # Project onto joint space
                            J = jacobian[:2, :]  # Take only position part
                            
                            # Compute modulation component
                            modulation_component = weight * np.outer(J.T @ e, J.T @ e)
                            
                            # Apply eigenvalue modulation
                            M_component = np.eye(3) - (1 - lambda1/lambda0) * modulation_component
                            
                            # Combine with current modulation matrix
                            M = M @ M_component
        
        return M

    def _compute_point_jacobian(self, state: JointState, segment_index: int, t: float) -> np.ndarray:
        """
        Compute Jacobian for a specific point along a robot segment.
        
        :param state: Joint state
        :param segment_index: Which segment (0, 1, or 2)
        :param t: Parameter along segment (0 at start, 1 at end)
        :return: 3x3 Jacobian matrix for the point
        """
        theta1, theta2, theta3 = state
        L1, L2, L3 = self.robot_link_lengths
        
        jacobian = np.zeros((3, 3))
        
        if segment_index == 0:  # First segment
            # Point along first link
            jacobian[0, 0] = -t * L1 * np.sin(theta1)
            jacobian[1, 0] = t * L1 * np.cos(theta1)
            
        elif segment_index == 1:  # Second segment
            # Point along second link
            jacobian[0, 0] = -L1 * np.sin(theta1) - t * L2 * np.sin(theta1 + theta2)
            jacobian[1, 0] = L1 * np.cos(theta1) + t * L2 * np.cos(theta1 + theta2)
            jacobian[0, 1] = -t * L2 * np.sin(theta1 + theta2)
            jacobian[1, 1] = t * L2 * np.cos(theta1 + theta2)
            
        else:  # Third segment
            # Point along third link
            jacobian[0, 0] = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2) - t * L3 * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 0] = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + t * L3 * np.cos(theta1 + theta2 + theta3)
            jacobian[0, 1] = -L2 * np.sin(theta1 + theta2) - t * L3 * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 1] = L2 * np.cos(theta1 + theta2) + t * L3 * np.cos(theta1 + theta2 + theta3)
            jacobian[0, 2] = -t * L3 * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 2] = t * L3 * np.cos(theta1 + theta2 + theta3)
        
        return jacobian

    def plan_dmm(self, start: JointState, goal: JointState, obstacles: list[ObstType],
                max_iter: int = 1000, step_size: float = 0.1,
                lambda0: float = 1.0, lambda1: float = 0.1,
                influence_distance: float = 1.0,
                attractive_gain: float = 1.0) -> PlanningResult:
        """
        Plan using Dynamic Modulation Matrix method.
        
        The DMM method modifies the motion field using a dynamic modulation matrix
        that ensures obstacle avoidance while maintaining stability.
        
        :param start: Start joint state
        :param goal: Goal joint state
        :param obstacles: List of obstacles
        :param max_iter: Maximum iterations
        :param step_size: Step size for motion
        :param lambda0: Base eigenvalue for free motion
        :param lambda1: Eigenvalue for obstacle avoidance
        :param influence_distance: Distance at which obstacles influence motion
        :param attractive_gain: Gain for attractive force
        :return: Planning result
        """
        path = [start]
        current_state = start
        
        for iteration in range(max_iter):
            # Check if we reached goal
            if self._distance(current_state, goal) < 0.1:
                path.append(goal)
                return True, path
            
            # Compute attractive force (towards goal)
            attractive_force = np.array([
                goal[i] - current_state[i] for i in range(3)
            ])
            attractive_force *= attractive_gain
            
            # Compute dynamic modulation matrix
            M = self._compute_dynamic_modulation_matrix(
                current_state, obstacles, lambda0, lambda1, influence_distance
            )
            
            # Apply modulation to the force
            modulated_force = M @ attractive_force
            
            # Normalize force
            force_norm = np.linalg.norm(modulated_force)
            if force_norm > 1.0:
                modulated_force = modulated_force / force_norm
            
            # Apply force to get new state
            new_state = tuple(
                current_state[i] + step_size * modulated_force[i] for i in range(3)
            )
            
            # Apply joint limits
            new_state = tuple(
                max(self.joint_limits[i][0], min(self.joint_limits[i][1], new_state[i]))
                for i in range(3)
            )
            
            # Check for collision
            if not self.collision_check(new_state, obstacles):
                current_state = new_state
                path.append(current_state)
            else:
                # If collision, try smaller step with original force
                smaller_step = step_size * 0.5
                new_state = tuple(
                    current_state[i] + smaller_step * attractive_force[i] for i in range(3)
                )
                new_state = tuple(
                    max(self.joint_limits[i][0], min(self.joint_limits[i][1], new_state[i]))
                    for i in range(3)
                )
                
                if not self.collision_check(new_state, obstacles):
                    current_state = new_state
                    path.append(current_state)
                else:
                    # Random perturbation to escape local minima
                    random_perturbation = tuple(
                        random.uniform(-0.1, 0.1) for _ in range(3)
                    )
                    new_state = tuple(
                        current_state[i] + random_perturbation[i] for i in range(3)
                    )
                    new_state = tuple(
                        max(self.joint_limits[i][0], min(self.joint_limits[i][1], new_state[i]))
                        for i in range(3)
                    )
                    
                    if not self.collision_check(new_state, obstacles):
                        current_state = new_state
                        path.append(current_state)
        
        return False, path

    def plan(
        self, start: JointState, goal: JointState, obstacles: list[ObstType],
        max_iter: int = 4000, step_size: float = 0.05,
        lambda0: float = 1.0, lambda1: float = 0.1,
        influence_distance: float = 1.0, attractive_gain: float = 1.0
    ) -> PlanningResult:
        """
        Plans a path from start Joint State to Goal Joint State using DMM algorithm.

        :param      start:                The start state
        :type       start:                JointState
        :param      goal:                 The goal state
        :type       goal:                 JointState
        :param      obstacles:            The obstacles
        :type       obstacles:            list[ObstType]
        :param      max_iter:             Maximum iterations
        :type       max_iter:             int
        :param      step_size:            Step size for motion
        :type       step_size:            float
        :param      lambda0:              Base eigenvalue for free motion
        :type       lambda0:              float
        :param      lambda1:              Eigenvalue for obstacle avoidance
        :type       lambda1:              float
        :param      influence_distance:   Distance at which obstacles influence motion
        :type       influence_distance:   float
        :param      attractive_gain:      Gain for attractive force
        :type       attractive_gain:      float

        :returns:   The planning result.
        :rtype:     PlanningResult
        """
        return self.plan_dmm(
            start, goal, obstacles, max_iter, step_size,
            lambda0, lambda1, influence_distance, attractive_gain
        )


if __name__ == "__main__":
    # Create a planner with robot link lengths
    link_lengths = [1.0, 0.8, 0.6]  # Three links of lengths 1.0, 0.8, 0.6
    planner = Planner(link_lengths)
    
    # Define start and goal configurations
    start_state = (0.0, 0.0, 0.0)  # All joints at 0 radians
    goal_state = (np.pi/2, -np.pi/4, np.pi/6)  # Target joint angles
    
    # Define obstacles (center_x, center_y), radius
    obstacles = [
        ((2.0, 1.2), 0.1),
        # ((0.5, -0.7), 0.1),
        # ((-0.5, 0.3), 0.1),
        # ((1.2, -0.8), 0.1)
    ]
    
    # Test DMM algorithm
    print(f"\n=== Testing DMM Algorithm ===")
    
    # Plan the path
    success, path = planner.plan(
        start_state, goal_state, obstacles, 
        max_iter=4000, 
        step_size=0.1,
        lambda0=2.0,
        lambda1=0.1,
        influence_distance=1.0,
        attractive_gain=2.0
    )
    
    # Print results
    print(f"Path planning {'SUCCESSFUL' if success else 'FAILED'}")
    if success:
        print(f"Path length: {len(path)} states")
        print(f"Start: {path[0]}")
        print(f"Goal: {path[-1]}")
        
        # Calculate path cost (sum of distances between consecutive states)
        path_cost = 0.0
        for i in range(len(path) - 1):
            path_cost += planner._distance(path[i], path[i+1])
        print(f"Path cost: {path_cost:.4f}")
        
        # Visualize the path
        fig, ax = plt.subplots(figsize=(12, 12))
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'DMM Algorithm\nPath Length: {len(path)}, Cost: {path_cost:.4f}')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        
        # Plot obstacles
        for center, radius in obstacles:
            circle = plt.Circle(center, radius, color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # Plot start and goal positions
        start_pos = planner.forward_kinematics(start_state)[-1]
        goal_pos = planner.forward_kinematics(goal_state)[-1]
        
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'bo', markersize=10, label='Goal')
        
        # Plot the path
        path_x = []
        path_y = []
        for state in path:
            ee_pos = planner.forward_kinematics(state)[-1]
            path_x.append(ee_pos[0])
            path_y.append(ee_pos[1])
        ax.plot(path_x, path_y, 'g-', alpha=0.7, linewidth=2, label='Path')
        
        # Plot final arm configuration
        final_positions = planner.forward_kinematics(path[-1])
        for i in range(len(final_positions) - 1):
            ax.plot([final_positions[i][0], final_positions[i+1][0]], 
                   [final_positions[i][1], final_positions[i+1][1]], 
                   'b-', linewidth=2, alpha=0.7)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Animation function
        def animate(frame):
            ax.clear()
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f'DMM Algorithm - Animation\nFrame {frame}/{len(path)}')
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            
            # Plot obstacles
            for center, radius in obstacles:
                circle = plt.Circle(center, radius, color='red', alpha=0.5)
                ax.add_patch(circle)
            
            # Plot start and goal
            start_pos = planner.forward_kinematics(start_state)[-1]
            goal_pos = planner.forward_kinematics(goal_state)[-1]
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
            ax.plot(goal_pos[0], goal_pos[1], 'bo', markersize=10, label='Goal')
            
            # Plot path up to current frame
            path_x = []
            path_y = []
            for i in range(min(frame + 1, len(path))):
                ee_pos = planner.forward_kinematics(path[i])[-1]
                path_x.append(ee_pos[0])
                path_y.append(ee_pos[1])
            ax.plot(path_x, path_y, 'g-', alpha=0.5, linewidth=2, label='Path')
            
            # Plot current arm configuration
            current_state = path[frame] if frame < len(path) else path[-1]
            positions = planner.forward_kinematics(current_state)
            
            # Draw arm segments
            for i in range(len(positions) - 1):
                ax.plot([positions[i][0], positions[i+1][0]], 
                       [positions[i][1], positions[i+1][1]], 
                       'b-', linewidth=3)
            
            # Draw joints
            for i, pos in enumerate(positions):
                color = 'red' if i == 0 else 'orange' if i < len(positions) - 1 else 'purple'
                marker = 's' if i == 0 else 'o'
                ax.plot(pos[0], pos[1], color=color, marker=marker, markersize=8)
            
            ax.legend()
            return ax
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(path), interval=150, repeat=False)
        plt.tight_layout()
        plt.show()
        
    else:
        print("No successful path found with DMM algorithm.")