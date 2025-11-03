#include "manipulator_controller.hpp"
#include <rclcpp/executors.hpp>
#include <thread>
#include <chrono>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<ManipulatorController>();
  
  // Wait for MoveGroup to initialize
  rclcpp::sleep_for(std::chrono::seconds(3));
  
  // Example 1: Move to a named target (if defined in SRDF)
  RCLCPP_INFO(node->get_logger(), "=== Example 1: Moving to named target ===");
  if (node->moveToNamedTarget("home"))  // Make sure this target exists in your SRDF
  {
    RCLCPP_INFO(node->get_logger(), "Successfully reached named target!");
  }
  else
  {
    RCLCPP_WARN(node->get_logger(), "Failed to reach named target, trying pose target instead...");
  }
  
  rclcpp::sleep_for(std::chrono::seconds(2));
  
  // Example 2: Move to specific pose (position + orientation)
  RCLCPP_INFO(node->get_logger(), "=== Example 2: Moving to target pose ===");
  // Adjust these coordinates based on your robot's workspace
  if (node->moveToPose(0.4, 0.1, 0.4, 0.0, M_PI/2, 0.0))  // x, y, z, roll, pitch, yaw
  {
    RCLCPP_INFO(node->get_logger(), "Successfully reached target pose!");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to reach target pose!");
  }
  
  rclcpp::sleep_for(std::chrono::seconds(2));
  
  // Example 3: Move to different position with current orientation
  RCLCPP_INFO(node->get_logger(), "=== Example 3: Moving to target position ===");
  if (node->moveToPosition(0.3, 0.2, 0.5))
  {
    RCLCPP_INFO(node->get_logger(), "Successfully reached target position!");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to reach target position!");
  }
  
  rclcpp::sleep_for(std::chrono::seconds(2));
  
  // Example 4: Another pose with different orientation
  RCLCPP_INFO(node->get_logger(), "=== Example 4: Moving to final pose ===");
  if (node->moveToPose(0.2, -0.1, 0.3, M_PI, 0.0, M_PI/4))
  {
    RCLCPP_INFO(node->get_logger(), "Successfully reached final pose!");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to reach final pose!");
  }
  
  RCLCPP_INFO(node->get_logger(), "=== Motion planning demo completed ===");
  
  rclcpp::shutdown();
  return 0;
}