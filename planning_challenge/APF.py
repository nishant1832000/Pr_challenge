"""
Artificial Potential Field Algorithm Implementation for 3-DOF Robotic Arm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from planner import Planner, JointState, ObstType

def run_apf_example():
    """Run APF algorithm with visualization"""
    # Create a planner with robot link lengths
    link_lengths = [1.0, 0.8, 0.6]
    planner = Planner(link_lengths)
    
    # Define start and goal configurations
    start_state: JointState = (0.0, 0.0, 0.0)
    goal_state: JointState = (np.pi/2, -np.pi/4, np.pi/6)
    
    # Define obstacles (simpler configuration for APF)
    obstacles: list[ObstType] = [
        ((1.75, 1.5), 0.1)
    ]
    
    print("=== Artificial Potential Field Algorithm ===")
    print("Planning path using potential fields...")
    
    # Plan using APF
    success, path = planner.plan(
        start_state, goal_state, obstacles,
        max_iter=1000,
        algorithm="APF",
        step_size=0.05,  # Smaller step for smoother APF motion
        apf_attractive_gain=1.0,
        apf_repulsive_gain=0.0
    )
    
    # Display results
    print(f"Path planning {'SUCCESSFUL' if success else 'FAILED'}")
    if success:
        print(f"Path length: {len(path)} states")
        
        # Calculate path cost
        path_cost = sum(planner._distance(path[i], path[i+1]) for i in range(len(path)-1))
        print(f"Path cost: {path_cost:.4f}")
        print("Note: APF produces smooth paths but can get stuck in local minima")
        
        # Create visualization
        visualize_apf_results(planner, start_state, goal_state, obstacles, path)
    else:
        print("No path found. APF may be stuck in local minima.")
        print("Try adjusting obstacle positions or algorithm parameters.")

def visualize_apf_results(planner: Planner, start: JointState, goal: JointState, 
                         obstacles: list[ObstType], path: list[JointState]):
    """Visualize APF results with plots and animation"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Final path and workspace
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('Artificial Potential Field - Smooth Path')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    
    # Plot obstacles
    for center, radius in obstacles:
        circle = plt.Circle(center, radius, color='red', alpha=0.5, label='Obstacle' if center == obstacles[0][0] else "")
        ax1.add_patch(circle)
    
    # Plot start and goal positions
    start_pos = planner.forward_kinematics(start)[-1]
    goal_pos = planner.forward_kinematics(goal)[-1]
    
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'bo', markersize=10, label='Goal')
    
    # Plot the complete path
    path_x = []
    path_y = []
    for state in path:
        ee_pos = planner.forward_kinematics(state)[-1]
        path_x.append(ee_pos[0])
        path_y.append(ee_pos[1])
    ax1.plot(path_x, path_y, 'g-', alpha=0.7, linewidth=2, label='Smooth Path')
    
    # Plot final arm configuration
    final_positions = planner.forward_kinematics(path[-1])
    for i in range(len(final_positions) - 1):
        ax1.plot([final_positions[i][0], final_positions[i+1][0]], 
                [final_positions[i][1], final_positions[i+1][1]], 
                'b-', linewidth=3, alpha=0.8)
    
    # Draw joints for final configuration
    for i, pos in enumerate(final_positions):
        color = 'red' if i == 0 else 'orange' if i < len(final_positions) - 1 else 'purple'
        marker = 's' if i == 0 else 'o'
        ax1.plot(pos[0], pos[1], color=color, marker=marker, markersize=8)
    
    ax1.legend()
    
    # Plot 2: Animation setup
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_title('APF - Animation')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    
    # Animation function
    def animate(frame):
        ax2.clear()
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_title(f'APF - Animation Frame {frame}/{len(path)}')
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        
        # Plot obstacles
        for center, radius in obstacles:
            circle = plt.Circle(center, radius, color='red', alpha=0.5)
            ax2.add_patch(circle)
        
        # Plot start and goal
        ax2.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax2.plot(goal_pos[0], goal_pos[1], 'bo', markersize=10, label='Goal')
        
        # Plot path up to current frame
        if frame > 0:
            current_path_x = path_x[:frame]
            current_path_y = path_y[:frame]
            ax2.plot(current_path_x, current_path_y, 'g-', alpha=0.5, linewidth=2, label='Smooth Path')
        
        # Plot current arm configuration
        current_state = path[frame] if frame < len(path) else path[-1]
        positions = planner.forward_kinematics(current_state)
        
        # Draw arm segments
        for i in range(len(positions) - 1):
            ax2.plot([positions[i][0], positions[i+1][0]], 
                   [positions[i][1], positions[i+1][1]], 
                   'b-', linewidth=3)
        
        # Draw joints
        for i, pos in enumerate(positions):
            color = 'red' if i == 0 else 'orange' if i < len(positions) - 1 else 'purple'
            marker = 's' if i == 0 else 'o'
            ax2.plot(pos[0], pos[1], color=color, marker=marker, markersize=8)
        
        ax2.legend()
        return ax2
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(path), interval=150, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_apf_example()