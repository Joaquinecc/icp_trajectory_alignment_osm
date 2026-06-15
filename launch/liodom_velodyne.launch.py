#!/usr/bin/env python3

import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution


def generate_launch_description():
    declare_map_points_filepath = DeclareLaunchArgument(
        "map_points_filepath",
        default_value="",
        description="Path to OSM points file (empty = auto-construct from frame_id)",
    )
    declare_bag_file = DeclareLaunchArgument(
        "bag_file", default_value="", description="Path to bag file"
    )
    declare_odom_topic_to_correct = DeclareLaunchArgument(
        "odom_topic_to_correct",
        default_value="/liodom/odom",
        # default_value='/Inertial_Labs/odom',
        description="Odom topic to correct",
    )
    declare_save_resuts_path = DeclareLaunchArgument(
        "save_resuts_path",
        default_value="/tmp/osm_align_results/",
        description="Save results path",
    )
    declare_gps_topic = DeclareLaunchArgument(
        "gps_topic", default_value="/kitti/oxts/gps", description="GPS topic"
    )
    declare_viz = DeclareLaunchArgument("viz", default_value="true", description="Viz")

    # Parameters for lane correction node
    declare_min_segment_size = DeclareLaunchArgument(
        "min_segment_size", default_value="150", description="Pose segment size"
    )
    declare_knn_neighbors = DeclareLaunchArgument(
        "knn_neighbors", default_value="20", description="KNN neighbors"
    )
    declare_icp_error_threshold = DeclareLaunchArgument(
        "icp_error_threshold", default_value="1.5", description="ICP error threshold"
    )
    declare_max_error_consecutive = DeclareLaunchArgument(
        "max_error_consecutive", default_value="50", description="Max error consecutive"
    )

    liodom_node = Node(
        package="liodom",
        executable="liodom_node",
        name="liodom",
        namespace="liodom",
        output="screen",
        respawn=False,
        parameters=[
            {
                "min_range": 3.0,
                "max_range": 50.0,
                "lidar_type": 0,
                "scan_lines": 64,
                "scan_regions": 8,
                "edges_per_region": 10,
                "prev_frames": 15,
                "fixed_frame": "odom",
                "base_frame": "base_link",
                "laser_frame": "velo_link",
                "use_imu": True,
                "save_results": False,
                "save_results_dir": "/tmp/save_results/",
                "mapping": False,
                "publish_tf": False,
                "use_sim_time": True,
            }
        ],
        remappings=[
            ("points", "/kitti/velo/pointcloud"),
            ("imu", "/kitti/oxts/imu"),
            ("map", "/liodom_mapper/map_local"),
        ],
    )

    odom_enu_correction_node = Node(
        package="osm_align",
        executable="odom_enu_correction_node",
        name="odom_enu_correction_node",
        output="screen",
        parameters=[
            {
                "odom_topic": LaunchConfiguration("odom_topic_to_correct"),
                "gps_topic": LaunchConfiguration("gps_topic"),
                "odom_output_topic": "/liodom/odom_enu",
            }
        ],
    )

    odometry_correction_node = Node(
        package="osm_align",
        executable="lane_correction_node",
        name="odometry_correction_node",
        output="screen",
        parameters=[
            {
                "map_points_filepath": LaunchConfiguration("map_points_filepath"),
                "odom_topic_to_correct": "/liodom/odom_enu",
                "initial_gps_topic": LaunchConfiguration("gps_topic"),
                "parameters_correction": ParameterValue(
                    [
                        TextSubstitution(text='{"min_segment_size": '),
                        LaunchConfiguration("min_segment_size"),
                        TextSubstitution(text=', "knn_neighbors": '),
                        LaunchConfiguration("knn_neighbors"),
                        TextSubstitution(text=', "icp_error_threshold": '),
                        LaunchConfiguration("icp_error_threshold"),
                        TextSubstitution(text=', "max_error_consecutive": '),
                        LaunchConfiguration("max_error_consecutive"),
                        TextSubstitution(text="}"),
                    ],
                    value_type=str,
                ),
                "save_resuts_path": LaunchConfiguration("save_resuts_path"),
            }
        ],
    )

    odom2gps_node = Node(
        package="osm_align",
        executable="odom2gps_node",
        name="odom2gps_node",
        output="screen",
        parameters=[
            {
                # 'odom_topic': "/osm_align/odom",
                "odom_topic": "/liodom/odom_enu",
                "gps_topic": LaunchConfiguration("gps_topic"),
            }
        ],
    )

    # Play ROS bag
    play_ros_bag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            LaunchConfiguration("bag_file"),
            "--delay",
            "1",
            "--rate",
            "1.0",
            "--start-offset",
            "0",
        ],
        # output='screen',
        condition=IfCondition(LaunchConfiguration("viz")),
    )

    # Rviz
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("osm_align"), "rviz", "liodom_kitti.rviz"
            ),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("viz")),
    )
    # Bridge socket
    bridge_socket = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "rosbridge_server",
            "rosbridge_websocket_launch.xml",
            "delay_between_messages:=0.0",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("viz")),
    )

    # TFs
    tf_base_link_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_lidar",
        arguments=["0", "0", "0", "0", "0", "0", "os_sensor", "base_link"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    tf_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_map_points_filepath,
            declare_save_resuts_path,
            declare_min_segment_size,
            declare_knn_neighbors,
            declare_icp_error_threshold,
            declare_odom_topic_to_correct,
            declare_max_error_consecutive,
            declare_viz,
            declare_bag_file,
            declare_gps_topic,
            # TFs
            tf_base_link_to_lidar,
            tf_map_to_odom,
            # Viz
            bridge_socket,
            rviz2,
            play_ros_bag,
            # Nodes
            liodom_node,
            odom_enu_correction_node,
            odometry_correction_node,
            odom2gps_node,
        ]
    )
