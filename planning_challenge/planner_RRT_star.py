import numpy as np
import heapq
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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

    # This Function randomly sample joint states
    def _random_sample(self) -> JointState:
        
        return tuple(random.uniform(low, high) for low, high in self.joint_limits)
    
    # This Function finds the nearest node in tree to the sample
    def _nearest_neighbor(self, tree: list, sample: JointState) -> JointState:
        
        return min(tree, key=lambda node: self._distance(node, sample))

    #########   THIS IS EXTRA FUNCTION FOR RRT*  #################
    # This function finds all nodes within certain radius of given state
    def _find_near_neighbors(self, tree: list, state: JointState, radius: float) -> list:
       
        return [node for node in tree if self._distance(node, state) <= radius]

    # Steer from from_state towards to_state with maximum step size
    def _steer(self, from_state: JointState, to_state: JointState, 
               max_step: float = 0.5) -> JointState:
        
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

    # This function checks if the path from state 1 to state 2 is collision free or not
    def _is_collision_free_path(self, state1: JointState, state2: JointState, 
                              obstacles: list[ObstType], num_checks: int = 5) -> bool:
        
        # Check intermediate points along the path
        for i in range(num_checks + 1):
            t = i / num_checks
            intermediate_state = tuple(
                state1[j] + t * (state2[j] - state1[j]) for j in range(3)
            )
            if self.collision_check(intermediate_state, obstacles):
                return False
        return True

    # Reconstruct the path by tracing the parent pointer from goal state to start state
    def _reconstruct_path(self, parent: dict, goal: JointState) -> list[JointState]:
       
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path

    #########   THIS IS EXTRA FUNCTION FOR RRT*  #################
    # This function calculate the path cost from current state to start state
    def _calculate_path_cost(self, parent: dict, state: JointState) -> float:
        
        cost = 0.0
        current = state
        while parent.get(current) is not None:
            cost += self._distance(parent[current], current)
            current = parent[current]
        return cost
     # This function plan a path by using RRT_STAR algorithm
    def plan(
        self, start: JointState, goal: JointState, obstacles: list[ObstType],
        max_iter: int = 4000, step_size: float = 0.3, goal_bias: float = 0.1,
        rrt_star_radius: float = 1.0
    ) -> PlanningResult:
        
        # Check if start or goal are in collision
        if self.collision_check(start, obstacles):
            return False, [start]
            
        if self.collision_check(goal, obstacles):
            return False, [start]

        # Initialize tree
        tree = [start]
        parent = {start: None}
        cost = {start: 0.0}     # Cost from start to each node

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
            if not self.collision_check(new_state, obstacles) and self._is_collision_free_path(nearest, new_state, obstacles):
                
                ######### RRT*: Find optimal parent and rewire tree  ########

                # Step 1: Find all near neighbors nodes
                near_nodes = self._find_near_neighbors(tree, new_state, rrt_star_radius)
                
                # Step 2: Find the node that gives minimum cost to reach new_state from start state
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
                
                # Check if we reached goal
                if self._distance(new_state, goal) <= step_size and self._is_collision_free_path(new_state, goal, obstacles):
                    
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
        ((1.5, 1.5), 0.15),
        ((0.5, -0.7), 0.1),
        ((-0.5, 0.3), 0.1),
        ((1.2, -0.8), 0.1)
    ]
    
    # Test RRT* algorithm
    print(f"\n=== Testing RRT* Algorithm ===")
    
    start_time = time.time()
    # Plan the path
    success, path = planner.plan(
        start_state, goal_state, obstacles, 
        max_iter=4000, 
        step_size=0.2,
        goal_bias=0.1,
        rrt_star_radius=0.5
    )

    time_taken = time.time() - start_time
    
    # Print results
    print(f"Path planning {'SUCCESSFUL' if success else 'FAILED'}")
    if success:
        print(f"Path length: {len(path)} states")
        print(f"Start: {path[0]}")
        print(f"Goal: {path[-1]}")
        print(f"Time taken for planning a path: {time_taken}")
        
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
        ax.set_title(f'RRT* Algorithm\nPath Length: {len(path)}, Cost: {path_cost:.4f}')
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
            ax.set_title(f'RRT* Algorithm - Animation\nFrame {frame}/{len(path)}')
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
        print("No successful path found with RRT* algorithm.")