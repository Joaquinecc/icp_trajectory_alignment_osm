#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    - transform_correction_service: Transform correction service name (default: '/liodom/transform_correction')
    - save_resuts_path: Directory to save results (poses.txt, runtime.txt). If empty, results are not saved.
    
    Example usage:
    ros2 launch osm_align osm_align.launch.py frame_id:=02 pose_segment_size:=100
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
    
    declare_pose_segment_size = DeclareLaunchArgument(
        'pose_segment_size', 
        default_value='150',
        description='Number of poses to maintain in sliding window buffer'
    )
    
    declare_knn_neighbors = DeclareLaunchArgument(
        'knn_neighbors',
        default_value='20', 
        description='Number of nearest neighbors for KD-tree spatial queries'
    )
    
    declare_valid_correspondence_threshold = DeclareLaunchArgument(
        'valid_correspondence_threshold',
        default_value='0.9',
        description='Minimum ratio of valid trajectory-to-map correspondences'
    )
    
    declare_icp_error_threshold = DeclareLaunchArgument(
        'icp_error_threshold',
        default_value='2.0',
        description='Maximum acceptable ICP alignment error threshold'
    )
    
    declare_trimming_ratio = DeclareLaunchArgument(
        'trimming_ratio',
        default_value='0.2',
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
    
    declare_transform_correction_service = DeclareLaunchArgument(
        'transform_correction_service',
        default_value='/liodom/transform_correction', 
        description='Transform correction service name for LIO-SAM integration'
    )

    declare_save_resuts_path = DeclareLaunchArgument(
        'save_resuts_path',
        default_value='/home/joaquinecc/Documents/ros_projects/results/osm_aligned/exp1/',
        description='Directory to save results (poses.txt, runtime.txt). If empty, results are not saved.'
    )
    
    # Create the node with parameters
    osm_align_node = Node(
        package='osm_align',
        executable='kitti_odometry',
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
            'transform_correction_service': LaunchConfiguration('transform_correction_service'),
            'save_resuts_path': LaunchConfiguration('save_resuts_path')
        }],
        remappings=[
            # Remap the odometry topic if specified
            ('/liodom/odom', LaunchConfiguration('odom_topic'))
        ]
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
        declare_transform_correction_service,
        declare_save_resuts_path,
        osm_align_node
    ]) 