// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sophus/interpolate.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

namespace {
Sophus::SE3d LookupTransform(const std::string &target_frame,
                             const std::string &source_frame,
                             const std::unique_ptr<tf2_ros::Buffer> &tf2_buffer) {
    std::string err_msg;
    if (tf2_buffer->canTransform(target_frame, source_frame, tf2::TimePointZero, &err_msg)) {
        try {
            auto tf = tf2_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "%s", ex.what());
        }
    }
    RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "Failed to find tf. Reason=%s",
                err_msg.c_str());
    // default construction is the identity
    return Sophus::SE3d();
}
}  // namespace

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    kiss_icp::pipeline::KISSConfig config;
    initializeParameters(config);

    // Construct the main KISS-ICP odometry node
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

    // Initialize subscribers
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));
    if (prior_source_ == "wheel_odom") {
        // BEST_EFFORT matches a RELIABLE or a BEST_EFFORT publisher; depth 50
        // rides out the odometry running several times the scan rate.
        odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            prior_odom_topic_, rclcpp::SensorDataQoS().keep_last(50),
            std::bind(&OdometryServer::OdometryCallback, this, std::placeholders::_1));
    }

    // Initialize publishers
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("kiss/odometry", qos);
    if (publish_debug_clouds_) {
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/frame", qos);
        kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/keypoints", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/local_map", qos);
    }

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf2_buffer_->setUsingDedicatedThread(true);
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);
    // Initialize service servers
    reset_service_ = create_service<std_srvs::srv::Empty>(
        "kiss/reset", std::bind(&OdometryServer::ResetService, this, std::placeholders::_1,
                                std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized");
}

void OdometryServer::initializeParameters(kiss_icp::pipeline::KISSConfig &config) {
    RCLCPP_INFO(this->get_logger(), "Initializing parameters");

    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    RCLCPP_INFO(this->get_logger(), "\tBase frame: %s", base_frame_.c_str());
    lidar_odom_frame_ = declare_parameter<std::string>("lidar_odom_frame", lidar_odom_frame_);
    RCLCPP_INFO(this->get_logger(), "\tLiDAR odometry frame: %s", lidar_odom_frame_.c_str());
    publish_odom_tf_ = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tPublish odometry transform: %d", publish_odom_tf_);
    invert_odom_tf_ = declare_parameter<bool>("invert_odom_tf", invert_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tInvert odometry transform: %d", invert_odom_tf_);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", publish_debug_clouds_);
    RCLCPP_INFO(this->get_logger(), "\tPublish debug clouds: %d", publish_debug_clouds_);
    position_covariance_ = declare_parameter<double>("position_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tPosition covariance: %.2f", position_covariance_);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tOrientation covariance: %.2f", orientation_covariance_);

    prior_source_ = declare_parameter<std::string>("prior.source", prior_source_);
    prior_odom_topic_ = declare_parameter<std::string>("prior.odom_topic", prior_odom_topic_);
    prior_max_age_ = declare_parameter<double>("prior.max_age", prior_max_age_);
    prior_rotation_only_ = declare_parameter<bool>("prior.rotation_only", prior_rotation_only_);
    if (prior_source_ != "constant_velocity" && prior_source_ != "wheel_odom") {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] prior.source '%s' is not one of constant_velocity | wheel_odom; "
                    "using constant_velocity",
                    prior_source_.c_str());
        prior_source_ = "constant_velocity";
    }
    RCLCPP_INFO(this->get_logger(), "\tMotion prior: %s", prior_source_.c_str());
    if (prior_source_ == "wheel_odom") {
        RCLCPP_INFO(this->get_logger(), "\t  odom topic: %s  max_age: %.2f s  rotation_only: %d",
                    prior_odom_topic_.c_str(), prior_max_age_, prior_rotation_only_);
    }

    config.max_range = declare_parameter<double>("data.max_range", config.max_range);
    RCLCPP_INFO(this->get_logger(), "\tMax range: %.2f", config.max_range);
    config.min_range = declare_parameter<double>("data.min_range", config.min_range);
    RCLCPP_INFO(this->get_logger(), "\tMin range: %.2f", config.min_range);
    config.deskew = declare_parameter<bool>("data.deskew", config.deskew);
    RCLCPP_INFO(this->get_logger(), "\tDeskew: %d", config.deskew);
    config.voxel_size = declare_parameter<double>("mapping.voxel_size", config.max_range / 100.0);
    RCLCPP_INFO(this->get_logger(), "\tVoxel size: %.2f", config.voxel_size);
    config.max_points_per_voxel =
        declare_parameter<int>("mapping.max_points_per_voxel", config.max_points_per_voxel);
    RCLCPP_INFO(this->get_logger(), "\tMax points per voxel: %d", config.max_points_per_voxel);
    config.initial_threshold =
        declare_parameter<double>("adaptive_threshold.initial_threshold", config.initial_threshold);
    RCLCPP_INFO(this->get_logger(), "\tInitial threshold: %.2f", config.initial_threshold);
    config.min_motion_th =
        declare_parameter<double>("adaptive_threshold.min_motion_th", config.min_motion_th);
    RCLCPP_INFO(this->get_logger(), "\tMin motion threshold: %.2f", config.min_motion_th);
    config.max_num_iterations =
        declare_parameter<int>("registration.max_num_iterations", config.max_num_iterations);
    RCLCPP_INFO(this->get_logger(), "\tMax number of iterations: %d", config.max_num_iterations);
    config.convergence_criterion = declare_parameter<double>("registration.convergence_criterion",
                                                             config.convergence_criterion);
    RCLCPP_INFO(this->get_logger(), "\tConvergence criterion: %.2f", config.convergence_criterion);
    config.max_num_threads =
        declare_parameter<int>("registration.max_num_threads", config.max_num_threads);
    RCLCPP_INFO(this->get_logger(), "\tMax number of threads: %d", config.max_num_threads);
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
        config.min_range = 0.0;
    }
}

void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    // Motion prior for this frame: the wheel-odometry delta since the previous
    // scan when configured and available, else nullopt (constant velocity).
    const double stamp = rclcpp::Time(msg->header.stamp).seconds();
    std::optional<Sophus::SE3d> prior;
    if (prior_source_ == "wheel_odom" && prev_scan_stamp_.has_value()) {
        prior = WheelPriorDelta(*prev_scan_stamp_, stamp, cloud_frame_id);
        if (prior.has_value()) {
            ++prior_used_;
        } else {
            ++prior_fallback_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "wheel prior unavailable for this scan (odometry stale or not "
                                 "bracketing %.3f..%.3f); using constant velocity. used=%zu "
                                 "fallback=%zu",
                                 *prev_scan_stamp_, stamp, prior_used_, prior_fallback_);
        }
    }
    prev_scan_stamp_ = stamp;

    // Register frame, main entry point to KISS-ICP pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps, prior);

    // Extract the last KISS-ICP pose, ego-centric to the LiDAR
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Spit the current estimated pose to ROS msgs handling the desired target frame
    PublishOdometry(kiss_pose, msg->header);
    // Publishing these clouds is a bit costly, so do it only if we are debugging
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header) {
    // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
    const auto cloud_frame_id = header.frame_id;
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto moving_frame = egocentric_estimation ? cloud_frame_id : base_frame_;
    const auto pose = [&]() -> Sophus::SE3d {
        if (egocentric_estimation) return kiss_pose;
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id, tf2_buffer_);
        return cloud2base * kiss_pose * cloud2base.inverse();
    }();

    // Broadcast the tf ---
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;
        if (invert_odom_tf_) {
            transform_msg.header.frame_id = moving_frame;
            transform_msg.child_frame_id = lidar_odom_frame_;
            transform_msg.transform = tf2::sophusToTransform(pose.inverse());
        } else {
            transform_msg.header.frame_id = lidar_odom_frame_;
            transform_msg.child_frame_id = moving_frame;
            transform_msg.transform = tf2::sophusToTransform(pose);
        }
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = lidar_odom_frame_;
    odom_msg.child_frame_id = moving_frame;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;
    odom_publisher_->publish(std::move(odom_msg));
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> &frame,
                                   const std::vector<Eigen::Vector3d> &keypoints,
                                   const std_msgs::msg::Header &header) {
    const auto kiss_map = kiss_icp_->LocalMap();

    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));
    auto local_map_header = header;
    local_map_header.frame_id = lidar_odom_frame_;
    map_publisher_->publish(std::move(EigenToPointCloud2(kiss_map, local_map_header)));
}
void OdometryServer::ResetService(
    [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    [[maybe_unused]] std::shared_ptr<std_srvs::srv::Empty::Response> response) {
    RCLCPP_INFO(this->get_logger(), "Resetting KISS-ICP map and odometry");

    // Reset the KISS-ICP pipeline
    kiss_icp_->Reset();
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        odom_buffer_.clear();
    }
    prev_scan_stamp_.reset();
    prior_used_ = prior_fallback_ = 0;

    RCLCPP_INFO(this->get_logger(), "KISS-ICP reset completed successfully");
}
void OdometryServer::OdometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
    const double stamp = rclcpp::Time(msg->header.stamp).seconds();
    const Sophus::SE3d pose = tf2::poseToSophus(msg->pose.pose);
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_child_frame_ = msg->child_frame_id;
    // Keep the buffer ordered and bounded: drop anything older than the window
    // needed to bracket the previous scan plus max_age.
    if (!odom_buffer_.empty() && stamp < odom_buffer_.back().first) return;  // out of order
    odom_buffer_.emplace_back(stamp, pose);
    const double keep_from = stamp - 2.0 * prior_max_age_ - 1.0;
    while (odom_buffer_.size() > 2 && odom_buffer_.front().first < keep_from) {
        odom_buffer_.pop_front();
    }
}

std::optional<Sophus::SE3d> OdometryServer::InterpolateOdometry(double stamp) const {
    // Caller holds odom_mutex_.
    if (odom_buffer_.size() < 2) return std::nullopt;
    // First sample at or after `stamp`.
    auto hi = std::lower_bound(
        odom_buffer_.begin(), odom_buffer_.end(), stamp,
        [](const std::pair<double, Sophus::SE3d> &s, double t) { return s.first < t; });
    if (hi == odom_buffer_.end()) {
        // Query is past the newest sample: allow a short extrapolation-free hold
        // only if the newest sample is fresh enough, else fail.
        const auto &last = odom_buffer_.back();
        return (stamp - last.first) <= prior_max_age_ ? std::optional(last.second) : std::nullopt;
    }
    if (hi == odom_buffer_.begin()) {
        return (hi->first - stamp) <= prior_max_age_ ? std::optional(hi->second) : std::nullopt;
    }
    const auto lo = std::prev(hi);
    const double span = hi->first - lo->first;
    if (span > prior_max_age_) return std::nullopt;  // gap in the odometry stream
    const double alpha = span > 0.0 ? (stamp - lo->first) / span : 0.0;
    return Sophus::interpolate(lo->second, hi->second, alpha);
}

std::optional<Sophus::SE3d> OdometryServer::WheelPriorDelta(double prev_stamp,
                                                            double curr_stamp,
                                                            const std::string &cloud_frame_id) {
    std::optional<Sophus::SE3d> p0, p1;
    std::string child_frame;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        p0 = InterpolateOdometry(prev_stamp);
        p1 = InterpolateOdometry(curr_stamp);
        child_frame = odom_child_frame_;
    }
    if (!p0.has_value() || !p1.has_value() || child_frame.empty()) return std::nullopt;

    // Motion of the odometry's child frame (its body, e.g. base_footprint)
    // between the two scans, in that body frame at prev_stamp:
    //   odom<-body(t1) = odom<-body(t0) * D_B
    const Sophus::SE3d delta_base = p0->inverse() * (*p1);

    // Rotate it into the sensor frame. With X = body<-sensor (static extrinsic),
    // odom<-sensor(t) = odom<-body(t) * X, so
    //   odom<-sensor(t1) = odom<-sensor(t0) * X^-1 * D_B * X
    // and X^-1 * D_B * X is the sensor-frame delta KissICP::delta() expects.
    // The extrinsic is between the ODOMETRY's body frame and the sensor, which
    // need not be base_frame_ (that only chooses the frame KISS reports in).
    const Sophus::SE3d X = LookupTransform(child_frame, cloud_frame_id, tf2_buffer_);
    Sophus::SE3d delta_sensor = X.inverse() * delta_base * X;

    if (prior_rotation_only_) {
        // Wheel yaw is usually trustworthy on this base; translation slips.
        // Keep the wheel rotation and the constant-velocity translation.
        delta_sensor.translation() = kiss_icp_->delta().translation();
    }
    return delta_sensor;
}

}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
