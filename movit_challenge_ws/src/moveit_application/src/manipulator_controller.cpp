#include "manipulator_controller.hpp"
#include <rclcpp/executors.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

using namespace std::chrono_literals;

ManipulatorController::ManipulatorController()
: Node("manipulator_controller")
{
  // Initialize MoveGroupInterface
  move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    shared_from_this(), planning_group_);
  
  // Set parameters
  move_group_->setMaxVelocityScalingFactor(0.5);  // 50% of maximum speed
  move_group_->setMaxAccelerationScalingFactor(0.5);
  move_group_->setPlanningTime(10.0);  // 10 seconds for planning
  move_group_->setNumPlanningAttempts(5);
  
  // Set end effector link
  move_group_->setEndEffectorLink(end_effector_link_);
  
  RCLCPP_INFO(this->get_logger(), "MoveGroupInterface initialized!");
  RCLCPP_INFO(this->get_logger(), "Planning frame: %s", move_group_->getPlanningFrame().c_str());
  RCLCPP_INFO(this->get_logger(), "End effector link: %s", move_group_->getEndEffectorLink().c_str());
  
  // Print current pose
  printCurrentPose();
}

bool ManipulatorController::moveToPose(const geometry_msgs::msg::Pose& target_pose)
{
  try
  {
    RCLCPP_INFO(this->get_logger(), "Planning to target pose...");
    
    // Display target pose for debugging
    displayTargetPose(target_pose);
    
    // Set the target pose
    move_group_->setPoseTarget(target_pose);
    
    // Plan and execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    
    bool success = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    
    if (success)
    {
      RCLCPP_INFO(this->get_logger(), "Planning successful! Executing trajectory...");
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "Execution completed!");
      
      // Print final pose
      printCurrentPose();
      return true;
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Planning failed!");
      return false;
    }
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "Exception in moveToPose: %s", e.what());
    return false;
  }
}

bool ManipulatorController::moveToPose(double x, double y, double z, double roll, double pitch, double yaw)
{
  geometry_msgs::msg::Pose target_pose = createPose(x, y, z, roll, pitch, yaw);
  return moveToPose(target_pose);
}

bool ManipulatorController::moveToPosition(double x, double y, double z)
{
  try
  {
    RCLCPP_INFO(this->get_logger(), "Moving to position [%.3f, %.3f, %.3f]", x, y, z);
    
    // Get current pose and only change position
    geometry_msgs::msg::Pose target_pose = move_group_->getCurrentPose().pose;
    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;
    
    return moveToPose(target_pose);
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "Exception in moveToPosition: %s", e.what());
    return false;
  }
}

bool ManipulatorController::moveToNamedTarget(const std::string& named_target)
{
  try
  {
    RCLCPP_INFO(this->get_logger(), "Moving to named target: %s", named_target.c_str());
    
    // Set named target (like "home", "ready", etc. defined in SRDF)
    move_group_->setNamedTarget(named_target);
    
    // Plan and execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    
    bool success = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    
    if (success)
    {
      RCLCPP_INFO(this->get_logger(), "Planning to '%s' successful! Executing...", named_target.c_str());
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "Execution completed!");
      return true;
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Planning to '%s' failed!", named_target.c_str());
      return false;
    }
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "Exception in moveToNamedTarget: %s", e.what());
    return false;
  }
}

geometry_msgs::msg::Pose ManipulatorController::createPose(double x, double y, double z, double roll, double pitch, double yaw)
{
  geometry_msgs::msg::Pose pose;
  
  // Set position
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  
  // Convert RPY to quaternion
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  pose.orientation = tf2::toMsg(q);
  
  return pose;
}

void ManipulatorController::printCurrentPose()
{
  try
  {
    geometry_msgs::msg::PoseStamped current_pose = move_group_->getCurrentPose();
    RCLCPP_INFO(this->get_logger(), "Current end effector pose:");
    RCLCPP_INFO(this->get_logger(), "  Position: [%.3f, %.3f, %.3f]", 
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z);
    RCLCPP_INFO(this->get_logger(), "  Orientation: [%.3f, %.3f, %.3f, %.3f]",
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w);
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(this->get_logger(), "Could not get current pose: %s", e.what());
  }
}

void ManipulatorController::displayTargetPose(const geometry_msgs::msg::Pose& pose)
{
  RCLCPP_INFO(this->get_logger(), "Target pose:");
  RCLCPP_INFO(this->get_logger(), "  Position: [%.3f, %.3f, %.3f]", 
              pose.position.x, pose.position.y, pose.position.z);
  RCLCPP_INFO(this->get_logger(), "  Orientation: [%.3f, %.3f, %.3f, %.3f]",
              pose.orientation.x, pose.orientation.y,
              pose.orientation.z, pose.orientation.w);
}