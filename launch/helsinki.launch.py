#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PythonExpression
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    declare_map_lanelet_path = DeclareLaunchArgument(
        'map_lanelet_path',
        default_value='',
        description='Path to OSM lanelet file (empty = auto-construct from frame_id)'
    )
    declare_bag_file = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to bag file'
    )
    declare_odom_topic = DeclareLaunchArgument(
        'odom_topic',
        default_value='/liodom/odom',
        description='Odom topic'
    )
    declare_pose_segment_size = DeclareLaunchArgument(
        'pose_segment_size',
        default_value='100',
        description='Pose segment size'
    )
    declare_knn_neighbors = DeclareLaunchArgument(
        'knn_neighbors',
        default_value='10',
        description='KNN neighbors'
    )
    declare_valid_correspondence_threshold = DeclareLaunchArgument(
        'valid_correspondence_threshold',
        default_value='0.9',
        description='Valid correspondence threshold'
    )
    declare_icp_error_threshold = DeclareLaunchArgument(
        'icp_error_threshold',
        default_value='2.0',
        description='ICP error threshold'
    )
    declare_trimming_ratio = DeclareLaunchArgument(
        'trimming_ratio',   
        default_value='0.2',
        description='Trimming ratio'
    )
    declare_min_distance_threshold = DeclareLaunchArgument(
        'min_distance_threshold',
        default_value='10.0',
        description='Min distance threshold'
    )
    declare_viz = DeclareLaunchArgument(
        'viz',
        default_value='True',
        description='Viz'
    )

    
    helsinki_node = Node(
        package='osm_align',
        executable='helsinki_node',
        name='helsinki_node',
        output='screen',
        parameters=[{
            'map_lanelet_path': LaunchConfiguration('map_lanelet_path'),
            'pose_segment_size': LaunchConfiguration('pose_segment_size'),
            'knn_neighbors': LaunchConfiguration('knn_neighbors'),
            'valid_correspondence_threshold': LaunchConfiguration('valid_correspondence_threshold'),
            'icp_error_threshold': LaunchConfiguration('icp_error_threshold'),
            'trimming_ratio': LaunchConfiguration('trimming_ratio'),
            'min_distance_threshold': LaunchConfiguration('min_distance_threshold'),
            'odom_topic': LaunchConfiguration('odom_topic'),
        }],
    )
    


    liodom_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("liodom"),
                "launch",
                "liodom_ouster_launch.xml",
            )
            
        ),
        launch_arguments={
            'viz': 'false',
            'mapping': 'false',
            'use_imu': 'true',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'liodom' in '", LaunchConfiguration('odom_topic'), "'"])
        )
    )

    # rviz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("osm_align"), "rviz", "helsinki.rviz")],
        output='screen',
        condition=IfCondition(LaunchConfiguration('viz'))  
    )

    play_ros_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file'), '--delay', '1', '--rate', '1.0','--start-offset', '0'],
        output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))

    )

    bridge_socket = ExecuteProcess(
        cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
        output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))
    )

    
    tf_base_link_to_lidar =Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_lidar',
            arguments=['0', '0', '0', '-1.5708', '0', '0', 'os_lidar', 'base_link'],
            parameters=[{'use_sim_time': True}], 
            output='screen'
    )
    tf_map_to_odom =Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{'use_sim_time': True}], 
            output='screen'
    )

    return LaunchDescription([
        declare_map_lanelet_path,
        declare_odom_topic,
        declare_pose_segment_size,
        declare_knn_neighbors,
        declare_valid_correspondence_threshold,
        declare_icp_error_threshold,
        declare_trimming_ratio,
        declare_min_distance_threshold,
        declare_viz,
        declare_bag_file,
        #Viz
        bridge_socket,
        rviz2 ,
        play_ros_bag,
        #TFs
        tf_base_link_to_lidar,
        tf_map_to_odom,
       #Nodes
        helsinki_node,
        liodom_launch,

    ])


