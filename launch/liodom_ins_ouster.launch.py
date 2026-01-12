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
from launch.substitutions import  TextSubstitution
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import numpy as np
from launch.substitutions import PythonExpression
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
        default_value='/liodom/odom_enu',
        # default_value='/Inertial_Labs/odom',
        description='Odom topic to correct'
    )
    declare_save_resuts_path = DeclareLaunchArgument(
        'save_resuts_path',
        default_value='/tmp/osm_align_results/',
        description='Save results path'
    )
    declare_viz = DeclareLaunchArgument(
        'viz',
        default_value='true',
        description='Viz'
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




    ins_conversion_node = Node(
        package='osm_align',
        executable='ins_conversion_node',
        name='ins_conversion_node',
        output='screen',
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
            'use_imu': 'false',
            'publish_tf': 'false',
        }.items(),
        condition=IfCondition(
            PythonExpression([
                "'liodom' in '",
                LaunchConfiguration('odom_topic_to_correct'),
                "'",
            ])
        ),
    )
    liodom_ouster_enu_corrector_node = Node(
        package='osm_align',
        executable='liodom_ouster_enu_corrector_node',
        name='liodom_ouster_enu_corrector_node',
        output='screen',
        condition=IfCondition(
          PythonExpression([
                "'liodom' in '",
                LaunchConfiguration('odom_topic_to_correct'),
                "'",
            ])
        ),
    )

    odometry_correction_node = Node(
        package='osm_align',
        executable='lane_correction_node',
        name='odometry_correction_node',
        output='screen',
        parameters=[{
            'map_points_filepath': LaunchConfiguration('map_points_filepath'),
            'odom_topic_to_correct': LaunchConfiguration('odom_topic_to_correct'),
            'initial_gps_topic': '/Inertial_Labs/initial_gps',
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

    odom2gps_node = Node(
        package='osm_align',
        executable='odom2gps_node',
        name='odom2gps_node',
        output='screen',
        parameters=[{
            'odom_topic': "/osm_align/odom",
            'gps_topic': '/Inertial_Labs/initial_gps',
        }],
    )


    #Play ROS bag
    play_ros_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file'), '--delay', '1', '--rate', '1.0','--start-offset', '0'],
        # output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))

    )
    
    #Rviz
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("osm_align"), "rviz", "liodom_ins_ouster.rviz")],
        output='screen',
        condition=IfCondition(LaunchConfiguration('viz'))  
    )
    #Bridge socket
    bridge_socket = ExecuteProcess(
        cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml', "delay_between_messages:=0.0"],
        output='screen',
         condition=IfCondition(LaunchConfiguration('viz'))
    )

    #TFs
    tf_base_link_to_lidar =Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_lidar',
            arguments=['0', '0', '0', "0", '0', '0', 'os_sensor', 'base_link'],
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
        declare_map_points_filepath,
        declare_save_resuts_path,
        declare_min_segment_size,
        declare_knn_neighbors,
        declare_icp_error_threshold,
        declare_odom_topic_to_correct,
        declare_max_error_consecutive,
        declare_viz,
        declare_bag_file,
        #TFs
        tf_base_link_to_lidar,
        tf_map_to_odom,
        #Viz
        bridge_socket,
        # rviz2 ,
        play_ros_bag,
       #Nodes
        ins_conversion_node,
        liodom_ouster_enu_corrector_node,
        odometry_correction_node,
        odom2gps_node,
        liodom_launch,

    ])


