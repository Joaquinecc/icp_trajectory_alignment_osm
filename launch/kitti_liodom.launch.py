#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PythonExpression, TextSubstitution
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import numpy as np  
def generate_launch_description():
    declare_map_points_filepath = DeclareLaunchArgument(
        'map_points_filepath',
        default_value='',
        description='Path to OSM points file (empty = auto-construct from frame_id)'
    )
    declare_bag_file = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to bag file'
    )
    declare_odom_topic_to_correct = DeclareLaunchArgument(
        'odom_topic_to_correct',
        default_value='/liodom/odom',
        # default_value='/Inertial_Labs/odom',
        description='Odom topic to correct'
    )
    declare_save_resuts_path = DeclareLaunchArgument(
        'save_resuts_path',
        default_value='/tmp/osm_align_results/',
        description='Save results path'
    )
    #Parameters for lane correction node
    declare_min_segment_size = DeclareLaunchArgument(
        'min_segment_size',
        default_value='150',
        description='Pose segment size'
    )
    declare_knn_neighbors = DeclareLaunchArgument(
        'knn_neighbors',
        default_value='20',
        description='KNN neighbors'
    )
    declare_icp_error_threshold = DeclareLaunchArgument(
        'icp_error_threshold',
        default_value='1.5',
        description='ICP error threshold'
    )
    declare_max_error_consecutive = DeclareLaunchArgument(
        'max_error_consecutive',
        default_value='50',
        description='Max error consecutive'
    )



    declare_viz = DeclareLaunchArgument(
        'viz',
        default_value='true',
        description='Viz'
    )

    liodom_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("liodom"),
                "launch",
                "liodom_launch.xml",
            )
        ),
        launch_arguments={
            'viz': 'false',
            'mapping': 'false',
            'use_imu': 'true',
        }.items(),
    )

    odometry_correction_node = Node(
        package='osm_align',
        executable='lane_correction_node',
        name='odometry_correction_node',
        output='screen',
        parameters=[{
            'map_points_filepath': LaunchConfiguration('map_points_filepath'),
            'odom_topic_to_correct': LaunchConfiguration('odom_topic_to_correct'),
            'initial_gps_topic': '/kitti/oxts/gps',
            'parameters_correction': ParameterValue([
                TextSubstitution(text='{"min_segment_size": '),
                LaunchConfiguration('min_segment_size'),
                TextSubstitution(text=', "knn_neighbors": '),
                LaunchConfiguration('knn_neighbors'),
                TextSubstitution(text=', "icp_error_threshold": '),
                LaunchConfiguration('icp_error_threshold'),
                TextSubstitution(text=', "max_error_consecutive": '),
                LaunchConfiguration('max_error_consecutive'),
                TextSubstitution(text='}'),
            ], value_type=str),
            'save_resuts_path': LaunchConfiguration('save_resuts_path'),
        }],
    )
    
        # rviz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("osm_align"), "rviz", "liodom_kitti.rviz")],
        output='screen',
        condition=IfCondition(LaunchConfiguration('viz'))  
    )
    

    play_ros_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file'), '--delay', '2', '--rate', '1.0','--start-offset', '0'],
        # output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))

    )


    tf_map_to_odom =Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=[
                '0', '0', '0',          # x y z
                '0', '0', '0',  # yaw pitch roll = -π/2, 0, -π/2
                'odom', 'map'
            ],
            parameters=[{'use_sim_time': True}], 
            output='screen'
    )    #Bridge socket
    bridge_socket = ExecuteProcess(
        cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml', "delay_between_messages:=0.0"],
        output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))
    )   

    

    return LaunchDescription([
        declare_map_points_filepath,
        declare_save_resuts_path,
        declare_min_segment_size,
        declare_knn_neighbors,
        declare_odom_topic_to_correct,
        declare_icp_error_threshold,
        declare_max_error_consecutive,
        declare_viz,
        declare_bag_file,
        #TFs
        tf_map_to_odom,
        #Viz
        bridge_socket,
        rviz2 ,
        play_ros_bag,
       #Nodes
        odometry_correction_node,
        liodom_launch,

    ])


