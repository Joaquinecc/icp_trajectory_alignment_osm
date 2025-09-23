#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Generate launch description for OSM alignment node.
    
    This launch file allows configuration of all parameters for the OSM alignment
    system including trajectory buffer size, ICP parameters, topic names, and
    KITTI sequence frame ID.
    
    Launch arguments:
    - frame_id: KITTI sequence identifier (default: '00')
    - map_lanelet_path: Path to OSM lanelet file (default: auto-constructed from frame_id)
    - pose_segment_size: Number of poses in sliding window buffer (default: 50)
    - knn_neighbors: Number of nearest neighbors for KD-tree queries (default: 10) 
    - valid_correspondence_threshold: Minimum ratio of valid correspondences (default: 0.6)
    - icp_error_threshold: Maximum ICP error for successful alignment (default: 1.5)
    - trimming_ratio: Trimming ratio for robust ICP (default: 0.2)
    - min_distance_threshold: Minimum trajectory distance for alignment (default: 10.0)
    - odom_topic: Input odometry topic name (default: '/liodom/odom')
    - save_resuts_path: Directory to save results (poses.txt, runtime.txt). If empty, results are not saved.
    - viz_marker_topic: Topic to publish lanelet markers (default: '/osm_align/lanelet_markers')
    - viz_frame: Frame for visualization markers (default: 'odom')
    - viz_line_width: Marker line width (default: 0.2)
    """
    
    # Declare launch arguments with default values
    declare_frame_id = DeclareLaunchArgument(
        'frame_id',
        default_value='00',
        description='KITTI sequence frame ID (e.g., 00, 01, 02, etc.)'
    )
    
    declare_map_lanelet_path = DeclareLaunchArgument(
        'map_lanelet_path',
        default_value='',
        description='Path to OSM lanelet file (empty = auto-construct from frame_id)'
    )
    declare_bag_path = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Path to KITTI bag file'
    )
    
    declare_pose_segment_size = DeclareLaunchArgument(
        'pose_segment_size', 
        default_value='100',
        description='Number of poses to maintain in sliding window buffer'
    )
    
    declare_knn_neighbors = DeclareLaunchArgument(
        'knn_neighbors',
        default_value='10', 
        description='Number of nearest neighbors for KD-tree spatial queries'
    )
    
    declare_valid_correspondence_threshold = DeclareLaunchArgument(
        'valid_correspondence_threshold',
        default_value='0.9',
        description='Minimum ratio of valid trajectory-to-map correspondences'
    )
    
    declare_icp_error_threshold = DeclareLaunchArgument(
        'icp_error_threshold',
        default_value='1.0',
        description='Maximum acceptable ICP alignment error threshold'
    )
    
    declare_trimming_ratio = DeclareLaunchArgument(
        'trimming_ratio',
        default_value='0.4',
        description='Trimming ratio for robust ICP algorithm'
    )
    
    declare_min_distance_threshold = DeclareLaunchArgument(
        'min_distance_threshold', 
        default_value='10.0',
        description='Minimum total trajectory distance before triggering alignment'
    )
    
    declare_odom_topic = DeclareLaunchArgument(
        'odom_topic',
        default_value='/liodom/odom',
        description='Input odometry topic name'
    )
    
    declare_save_resuts_path = DeclareLaunchArgument(
        'save_resuts_path',
        default_value='~/temp/',
        description='Directory to save results (poses.txt, runtime.txt). If empty, results are not saved.'
    )

    # Viz arguments
    declare_viz_marker_topic = DeclareLaunchArgument(
        'viz_marker_topic',
        default_value='/osm_align/lanelet_markers',
        description='Topic to publish lanelet markers'
    )
    declare_viz_frame = DeclareLaunchArgument(
        'viz_frame',
        default_value='odom',
        description='Frame for visualization markers'
    )
    declare_viz_line_width = DeclareLaunchArgument(
        'viz_line_width',
        default_value='0.2',
        description='Line width for visualization markers'
    )
    
    # Create the node with parameters
    osm_align_node = Node(
        package='osm_align',
        executable='kitti_node',
        name='osm_align_node',
        output='screen',
        parameters=[{
            # 'frame_id': LaunchConfiguration('frame_id'),
            'frame_id': ParameterValue(LaunchConfiguration('frame_id'), value_type=str),
            'map_lanelet_path': LaunchConfiguration('map_lanelet_path'),
            'pose_segment_size': LaunchConfiguration('pose_segment_size'),
            'knn_neighbors': LaunchConfiguration('knn_neighbors'), 
            'valid_correspondence_threshold': LaunchConfiguration('valid_correspondence_threshold'),
            'icp_error_threshold': LaunchConfiguration('icp_error_threshold'),
            'trimming_ratio': LaunchConfiguration('trimming_ratio'),
            'min_distance_threshold': LaunchConfiguration('min_distance_threshold'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'save_resuts_path': LaunchConfiguration('save_resuts_path')
        }],
        remappings=[
            # Remap the odometry topic if specified
            ('/liodom/odom', LaunchConfiguration('odom_topic'))
        ]
    )

    liodom_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("liodom"),
                    "launch",
                    "liodom_launch.xml",    
                )
            ]
        ),
        launch_arguments={'viz': 'false', 'mapping': 'false', 'use_imu': 'false'}.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_node',
        output='screen',
        parameters=[{'config': '${find:osm_align}/rviz/kitti.rviz'}]
    )
    
    kitti_bag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'),
             '--clock',
             '--delay', '1.5',
             '--wait-for-all-acked', '0',
             '--rate', '1.0',
            #  '--qos-profile-overrides-path', '${find:kitti_to_ros2bag}/config/qos_override.yaml'
             ],
        # output='screen'
    )
    osm_map_viz_node = Node(
        package='osm_align',
        executable='map_osm_viz',
        name='osm_map_viz_node',
        output='screen',
        parameters=[{"gps_topic":"/kitti/oxts/gps",
                    'odom_topic1':"/kitti/gtruth/odom",
                     'odom_topic2':"/liodom/odom",
                     'odom_topic3':"/osm_align/odom_aligned"}]
    )

    rosbridge_node = ExecuteProcess(
        cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
        output='screen'
    )

    return LaunchDescription([  
        declare_frame_id,
        declare_map_lanelet_path,
        declare_pose_segment_size,
        declare_knn_neighbors,
        declare_valid_correspondence_threshold,
        declare_icp_error_threshold,
        declare_trimming_ratio,
        declare_min_distance_threshold,
        declare_odom_topic,
        declare_save_resuts_path,
        declare_viz_marker_topic,
        declare_viz_frame,
        declare_viz_line_width,
        osm_align_node,
        liodom_launch,    
        osm_map_viz_node,
        rosbridge_node,
        rviz_node,
        kitti_bag_node,
    ]) 