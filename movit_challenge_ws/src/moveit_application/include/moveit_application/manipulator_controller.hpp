#ifndef MANIPULATOR_CONTROLLER_HPP
#define MANIPULATOR_CONTROLLER_HPP

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class ManipulatorController : public rclcpp::Node
{
public:
  ManipulatorController();
  bool moveToPose(const geometry_msgs::msg::Pose& target_pose);
  bool moveToPose(double x, double y, double z, double roll, double pitch, double yaw);
  bool moveToPosition(double x, double y, double z);
  bool moveToNamedTarget(const std::string& named_target);
  void displayTargetPose(const geometry_msgs::msg::Pose& pose);

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::string planning_group_ = "ur5e_arm";
  std::string end_effector_link_ = "tool0";  // Adjust based on your URDF

  geometry_msgs::msg::Pose createPose(double x, double y, double z, double roll, double pitch, double yaw);
  void printCurrentPose();
};

#endif