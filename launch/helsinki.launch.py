#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
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
            'mapping': 'true',
            'use_imu': 'false',
        }.items()
    )
    helsinki_node = Node(
        package='osm_align',
        executable='helsinki_node',
        name='helsinki_node',
        output='screen',
        parameters=[{
            'map_lanelet_path': LaunchConfiguration('map_lanelet_path'),
        }],
    )

        # rviz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("osm_align"), "rviz", "helsinki.rviz")],
        output='screen'
    )

    play_ros_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file'), '--delay', '5', '--rate', '0.8'],
        output='screen'
    )
    return LaunchDescription([
        declare_map_lanelet_path,
        declare_bag_file,
        liodom_launch,
        helsinki_node,
        rviz2,
        play_ros_bag,
    ])


