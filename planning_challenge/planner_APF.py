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

    # Calculate Each joint positions in cartesian space   
    def forward_kinematics(self, state: JointState) -> list:
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

    # Check the collision of line segemnt and circle
    def collision_check(self, state: JointState, obstacles: list[ObstType]) -> bool:
        
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
    
    # This function checks if the line segment is collide with circle or not
    def _segment_circle_collision(self, p1: tuple, p2: tuple, center: tuple, radius: float) -> bool:
       
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

    # This function calculate the distance between joint state 1 and joint state 2 
    def _distance(self, state1: JointState, state2: JointState) -> float:
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(state1, state2)))

    # This function Compute attractive and repulsive forces for APF.
    def _compute_apf_forces(self, state: JointState, goal: JointState, 
                           obstacles: list[ObstType], 
                           attractive_gain: float = 1.0, 
                           repulsive_gain: float = 0.5,
                           repulsive_range: float = 1.0) -> tuple:
        
        # Attractive force (towards goal)
        attractive_force = np.array([
            goal[i] - state[i] for i in range(3)
        ])
        attractive_force *= attractive_gain
        
        # Repulsive force (away from obstacles)
        repulsive_force = np.zeros(3)
        positions = self.forward_kinematics(state)
        
        # Check each arm segment for proximity to obstacles
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
                    
                    distance = np.sqrt((closest_x - center[0])**2 + (closest_y - center[1])**2)
                    
                    # Only consider obstacles within repulsive range
                    if distance < repulsive_range + radius:
                        # Avoid division by zero
                        safe_distance = max(distance - radius, 0.001)
                        
                        if safe_distance < repulsive_range:
                            # Compute repulsive force magnitude
                            force_magnitude = repulsive_gain * (1/safe_distance - 1/repulsive_range) * (1/safe_distance**2)
                            
                            # Direction away from obstacle (in workspace)
                            if distance > radius:
                                dir_x = (closest_x - center[0]) / (distance + 1e-8)
                                dir_y = (closest_y - center[1]) / (distance + 1e-8)
                            else:
                                # If inside obstacle, push in random direction
                                dir_x = random.uniform(-1, 1)
                                dir_y = random.uniform(-1, 1)
                                norm = np.sqrt(dir_x**2 + dir_y**2)
                                dir_x /= norm
                                dir_y /= norm
                            
                            # Convert workspace force to joint space (simplified)
                            # This is a simplified Jacobian-based approach
                            jacobian = self._compute_simplified_jacobian(state, i)
                            joint_force = np.dot(jacobian.T, [dir_x * force_magnitude, dir_y * force_magnitude])
                            repulsive_force += joint_force
        
        total_force = attractive_force + repulsive_force
        
        # Normalize force to prevent large steps
        force_norm = np.linalg.norm(total_force)
        if force_norm > 1.0:
            total_force = total_force / force_norm
        
        return total_force

    # Compute a simplified Jacobian for force transformation from workspace to joint space.
    def _compute_simplified_jacobian(self, state: JointState, segment_index: int) -> np.ndarray:
        
        theta1, theta2, theta3 = state
        jacobian = np.zeros((2, 3))
        
        if segment_index == 0:  # First link
            jacobian[0, 0] = -self.robot_link_lengths[0] * np.sin(theta1)
            jacobian[1, 0] = self.robot_link_lengths[0] * np.cos(theta1)
            
        elif segment_index == 1:  # Second link
            jacobian[0, 0] = -self.robot_link_lengths[0] * np.sin(theta1) - self.robot_link_lengths[1] * np.sin(theta1 + theta2)
            jacobian[1, 0] = self.robot_link_lengths[0] * np.cos(theta1) + self.robot_link_lengths[1] * np.cos(theta1 + theta2)
            jacobian[0, 1] = -self.robot_link_lengths[1] * np.sin(theta1 + theta2)
            jacobian[1, 1] = self.robot_link_lengths[1] * np.cos(theta1 + theta2)
            
        else:  # Third link
            jacobian[0, 0] = -self.robot_link_lengths[0] * np.sin(theta1) - self.robot_link_lengths[1] * np.sin(theta1 + theta2) - self.robot_link_lengths[2] * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 0] = self.robot_link_lengths[0] * np.cos(theta1) + self.robot_link_lengths[1] * np.cos(theta1 + theta2) + self.robot_link_lengths[2] * np.cos(theta1 + theta2 + theta3)
            jacobian[0, 1] = -self.robot_link_lengths[1] * np.sin(theta1 + theta2) - self.robot_link_lengths[2] * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 1] = self.robot_link_lengths[1] * np.cos(theta1 + theta2) + self.robot_link_lengths[2] * np.cos(theta1 + theta2 + theta3)
            jacobian[0, 2] = -self.robot_link_lengths[2] * np.sin(theta1 + theta2 + theta3)
            jacobian[1, 2] = self.robot_link_lengths[2] * np.cos(theta1 + theta2 + theta3)
        
        return jacobian

    # This function plan using ARTIFICIAL POTENTIAL FIELD
    def plan_apf(self, start: JointState, goal: JointState, obstacles: list[ObstType],
                max_iter: int = 1000, step_size: float = 0.1,
                attractive_gain: float = 1.0, repulsive_gain: float = 0.5,
                repulsive_range: float = 1.0) -> PlanningResult:
       
        path = [start]
        current_state = start
        
        for iteration in range(max_iter):
            # Check if we reached goal
            if self._distance(current_state, goal) < 0.1:
                path.append(goal)
                return True, path
            
            # Compute forces
            force = self._compute_apf_forces(
                current_state, goal, obstacles, 
                attractive_gain, repulsive_gain, repulsive_range
            )
            
            # Apply force to get new state
            new_state = tuple(
                current_state[i] + step_size * force[i] for i in range(3)
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
                # If collision, try smaller step or random walk
                smaller_step = step_size * 0.5
                new_state = tuple(
                    current_state[i] + smaller_step * force[i] for i in range(3)
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


if __name__ == "__main__":
    # Create a planner with robot link lengths
    link_lengths = [1.0, 0.8, 0.6]  # Three links of lengths 1.0, 0.8, 0.6
    planner = Planner(link_lengths)
    
    # Define start and goal configurations
    start_state = (0.0, 0.0, 0.0)  # All joints at 0 radians
    goal_state = (np.pi/2, -np.pi/4, np.pi/6)  # Target joint angles
    
    # Define obstacles (center_x, center_y), radius
    obstacles = [
        ((2.0, 1.5), 0.15),
        ((0.5, -0.7), 0.1),
        ((-0.5, 0.3), 0.1),
        ((1.2, -0.8), 0.1)
    ]
    
    # Test APF algorithm
    print(f"\n=== Testing APF Algorithm ===")
    
    # Plan the path
    success, path = planner.plan_apf(
        start_state, goal_state, obstacles, 
        max_iter=10000, 
        step_size=0.05,
        attractive_gain=5.0,
        repulsive_gain=0.03
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
        ax.set_title(f'APF Algorithm\nPath Length: {len(path)}, Cost: {path_cost:.4f}')
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
            ax.set_title(f'APF Algorithm - Animation\nFrame {frame}/{len(path)}')
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
        print("No successful path found with APF algorithm.")
        