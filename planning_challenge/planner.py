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

    def _random_sample(self) -> JointState:
        """
        Generate a random sample in joint space.
        """
        return tuple(random.uniform(low, high) for low, high in self.joint_limits)

    def _nearest_neighbor(self, tree: list, sample: JointState) -> JointState:
        """
        Find the nearest node in the tree to the sample.
        """
        return min(tree, key=lambda node: self._distance(node, sample))

    def _find_near_neighbors(self, tree: list, state: JointState, radius: float) -> list:
        """
        Find all nodes within a certain radius of the given state.
        """
        return [node for node in tree if self._distance(node, state) <= radius]

    def _steer(self, from_state: JointState, to_state: JointState, 
               max_step: float = 0.5) -> JointState:
        """
        Steer from from_state towards to_state with maximum step size.
        """
        dist = self._distance(from_state, to_state)
        if dist <= max_step:
            return to_state
            
        # Interpolate
        ratio = max_step / dist
        new_state = tuple(
            from_state[i] + ratio * (to_state[i] - from_state[i])
            for i in range(3)
        )
        return new_state

    def _is_collision_free_path(self, state1: JointState, state2: JointState, 
                              obstacles: list[ObstType], num_checks: int = 5) -> bool:
        """
        Check if the path between two states is collision-free.
        """
        # Check intermediate points along the path
        for i in range(num_checks + 1):
            t = i / num_checks
            intermediate_state = tuple(
                state1[j] + t * (state2[j] - state1[j]) for j in range(3)
            )
            if self.collision_check(intermediate_state, obstacles):
                return False
        return True

    def _reconstruct_path(self, parent: dict, goal: JointState) -> list[JointState]:
        """
        Reconstruct the path from start to goal using parent pointers.
        """
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path

    def _calculate_path_cost(self, parent: dict, state: JointState) -> float:
        """
        Calculate the cost from start to the given state.
        """
        cost = 0.0
        current = state
        while parent.get(current) is not None:
            cost += self._distance(parent[current], current)
            current = parent[current]
        return cost

    def _compute_apf_forces(self, state: JointState, goal: JointState, 
                           obstacles: list[ObstType], 
                           attractive_gain: float = 1.0, 
                           repulsive_gain: float = 0.5,
                           repulsive_range: float = 1.0) -> tuple:
        """
        Compute attractive and repulsive forces for APF.
        Returns total force as a 3D vector in joint space.
        """
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

    def _compute_simplified_jacobian(self, state: JointState, segment_index: int) -> np.ndarray:
        """
        Compute a simplified Jacobian for force transformation from workspace to joint space.
        """
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

    def plan_apf(self, start: JointState, goal: JointState, obstacles: list[ObstType],
                max_iter: int = 1000, step_size: float = 0.1,
                attractive_gain: float = 1.0, repulsive_gain: float = 0.5,
                repulsive_range: float = 1.0) -> PlanningResult:
        """
        Plan using Artificial Potential Field method.
        """
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

    def plan(
        self, start: JointState, goal: JointState, obstacles: list[ObstType],
        max_iter: int = 1000, step_size: float = 0.3, goal_bias: float = 0.1,
        algorithm: str = "RRT", rrt_star_radius: float = 1.0,
        apf_attractive_gain: float = 1.0, apf_repulsive_gain: float = 0.5
    ) -> PlanningResult:
        """
        Plans a path from start Joint State to Goal Joint State using various algorithms.

        :param      start:                The start state
        :type       start:                JointState
        :param      goal:                 The goal state
        :type       goal:                 JointState
        :param      obstacles:            The obstacles
        :type       obstacles:            list[ObstType]
        :param      max_iter:             Maximum iterations
        :type       max_iter:             int
        :param      step_size:            Maximum step size for steering
        :type       step_size:            float
        :param      goal_bias:            Probability of sampling the goal directly
        :type       goal_bias:            float
        :param      algorithm:            "RRT", "RRT*", or "APF"
        :type       algorithm:            str
        :param      rrt_star_radius:      Radius for near neighbor search in RRT*
        :type       rrt_star_radius:      float
        :param      apf_attractive_gain:  Attractive force gain for APF
        :type       apf_attractive_gain:  float
        :param      apf_repulsive_gain:   Repulsive force gain for APF
        :type       apf_repulsive_gain:   float

        :returns:   The planning result.
        :rtype:     PlanningResult
        """
        if algorithm == "APF":
            return self.plan_apf(
                start, goal, obstacles, max_iter, step_size,
                apf_attractive_gain, apf_repulsive_gain
            )
        
        # Check if start or goal are in collision
        if self.collision_check(start, obstacles):
            return False, [start]
            
        if self.collision_check(goal, obstacles):
            return False, [start]

        # Initialize tree
        tree = [start]
        parent = {start: None}
        cost = {start: 0.0}  # Only used for RRT*

        for iteration in range(max_iter):
            # Sample with goal bias
            if random.random() < goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            # Find nearest node in tree
            nearest = self._nearest_neighbor(tree, sample)
            
            # Steer towards sample
            new_state = self._steer(nearest, sample, step_size)
            
            # Check if path from nearest to new_state is collision-free
            if not self.collision_check(new_state, obstacles) and \
               self._is_collision_free_path(nearest, new_state, obstacles):
                
                if algorithm == "RRT":
                    # Standard RRT: just add the new node
                    tree.append(new_state)
                    parent[new_state] = nearest
                    cost[new_state] = cost[nearest] + self._distance(nearest, new_state)
                    
                elif algorithm == "RRT*":
                    # RRT*: Find optimal parent and rewire tree
                    # Step 1: Find near neighbors
                    near_nodes = self._find_near_neighbors(tree, new_state, rrt_star_radius)
                    
                    # Step 2: Find the node that gives minimum cost to reach new_state
                    min_cost = cost[nearest] + self._distance(nearest, new_state)
                    best_parent = nearest
                    
                    for node in near_nodes:
                        if node != nearest:
                            # Check if path from node to new_state is collision-free
                            if self._is_collision_free_path(node, new_state, obstacles):
                                new_cost = cost[node] + self._distance(node, new_state)
                                if new_cost < min_cost:
                                    min_cost = new_cost
                                    best_parent = node
                    
                    # Add new node to tree with best parent
                    tree.append(new_state)
                    parent[new_state] = best_parent
                    cost[new_state] = min_cost
                    
                    # Step 3: Rewire the tree
                    for node in near_nodes:
                        if node != best_parent:
                            # Check if going through new_state gives lower cost for node
                            new_cost_to_node = cost[new_state] + self._distance(new_state, node)
                            if new_cost_to_node < cost[node]:
                                # Check if path is collision-free
                                if self._is_collision_free_path(new_state, node, obstacles):
                                    # Rewire: make new_state the parent of node
                                    parent[node] = new_state
                                    cost[node] = new_cost_to_node
                
                # Check if we reached goal (for both algorithms)
                if self._distance(new_state, goal) <= step_size and \
                   self._is_collision_free_path(new_state, goal, obstacles):
                    
                    # Connect to goal
                    if algorithm == "RRT":
                        parent[goal] = new_state
                        cost[goal] = cost[new_state] + self._distance(new_state, goal)
                    else:  # RRT*
                        # For RRT*, find optimal connection to goal
                        near_goal_nodes = self._find_near_neighbors(tree, goal, rrt_star_radius)
                        min_goal_cost = float('inf')
                        best_goal_parent = None
                        
                        for node in near_goal_nodes:
                            if self._is_collision_free_path(node, goal, obstacles):
                                goal_cost = cost[node] + self._distance(node, goal)
                                if goal_cost < min_goal_cost:
                                    min_goal_cost = goal_cost
                                    best_goal_parent = node
                        
                        if best_goal_parent is not None:
                            parent[goal] = best_goal_parent
                            cost[goal] = min_goal_cost
                    
                    if goal in parent:  # Goal was successfully connected
                        path = self._reconstruct_path(parent, goal)
                        return True, path

        # No path found within max iterations
        return False, [start]


if __name__ == "__main__":
    # Create a planner with robot link lengths
    link_lengths = [1.0, 0.8, 0.6]  # Three links of lengths 1.0, 0.8, 0.6
    planner = Planner(link_lengths)
    
    # Define start and goal configurations
    start_state = (0.0, 0.0, 0.0)  # All joints at 0 radians
    goal_state = (np.pi/2, -np.pi/4, np.pi/6)  # Target joint angles
    
    # Define obstacles (center_x, center_y), radius
    obstacles = [
        ((1.0, 0.5), 0.1),
        ((0.5, -0.7), 0.1),
        ((-0.5, 0.3), 0.1),
        ((1.2, -0.8), 0.1)
    ]
    
    # Test all algorithms
    algorithms = ["RRT", "RRT*", "APF"]
    results = {}
    
    for algo in algorithms:
        print(f"\n=== Testing {algo} Algorithm ===")
        
        # Plan the path
        if algo == "APF":
            success, path = planner.plan(
                start_state, goal_state, obstacles, 
                max_iter=1000, 
                algorithm=algo,
                step_size=0.05,  # Smaller step for APF
                apf_attractive_gain=1.0,
                apf_repulsive_gain=0.8
            )
        else:
            success, path = planner.plan(
                start_state, goal_state, obstacles, 
                max_iter=2000, 
                algorithm=algo,
                rrt_star_radius=1.0
            )
        
        # Store results
        results[algo] = {
            'success': success,
            'path': path,
            'path_length': len(path) if success else 0
        }
        
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
    
    # Compare results
    print(f"\n=== Algorithm Comparison ===")
    for algo in algorithms:
        result = results[algo]
        if result['success']:
            path_cost = sum(planner._distance(result['path'][i], result['path'][i+1]) 
                          for i in range(len(result['path']) - 1))
            print(f"{algo}: Path found with {len(result['path'])} states, cost: {path_cost:.4f}")
        else:
            print(f"{algo}: No path found")
    
    # Visualize all successful paths
    successful_algorithms = [algo for algo in algorithms if results[algo]['success']]
    
    if successful_algorithms:
        # Create subplots for comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()
        
        # Plot each successful algorithm
        for idx, algo in enumerate(successful_algorithms):
            if idx >= 4:  # Maximum 4 subplots
                break
                
            ax = axes[idx]
            path = results[algo]['path']
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f'{algo} Algorithm\nPath Length: {len(path)}, Cost: {sum(planner._distance(path[i], path[i+1]) for i in range(len(path)-1)):.4f}')
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
        
        # Hide unused subplots
        for idx in range(len(successful_algorithms), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Create animation for the best path (lowest cost)
        best_algo = None
        best_cost = float('inf')
        
        for algo in successful_algorithms:
            path = results[algo]['path']
            path_cost = sum(planner._distance(path[i], path[i+1]) for i in range(len(path)-1))
            if path_cost < best_cost:
                best_cost = path_cost
                best_algo = algo
        
        if best_algo:
            print(f"\nAnimating best path from {best_algo} with cost {best_cost:.4f}")
            success, path = results[best_algo]['success'], results[best_algo]['path']
            
            # Create animation
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Animation function
            def animate(frame):
                ax.clear()
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_title(f'{best_algo} Algorithm - Animation\nFrame {frame}/{len(path)}')
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
            anim = FuncAnimation(fig, animate, frames=len(path), interval=200, repeat=True)
            plt.tight_layout()
            plt.show()
            
    else:
        print("No successful path found with any algorithm.")