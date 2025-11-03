from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur5e", package_name="ur5e_arm_moveit_config").to_moveit_configs()

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "rviz", "view_ur5e.rviz"]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    # Generate the base demo launch description
    demo_launch = generate_demo_launch(moveit_config)

    # Now return a LaunchDescription including both demo_launch and rviz_node
    return LaunchDescription(demo_launch.entities + [rviz_node])
    
