import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import math
from typing import List, Tuple, Optional
import os
import lanelet2
from geometry_msgs.msg import Transform
from liodom.srv import TransformCorrection
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose
import osm_align.utils as utils
from osm_align.kitti_utils import angle_dict, cordinta_dict
import argparse
import sys

# Configuration parameters are now declared as ROS parameters in the node


class OdomCorrection(Node):
    """
    ROS2 node for real-time odometry correction using lanelet map alignment.

    Subscribes to odometry messages, maintains a sliding window of poses, and
    periodically aligns the trajectory to a lanelet map using robust ICP algorithms.
    When alignment is successful, sends transform corrections to the LIO-SAM system.

    This node implements trajectory-to-map alignment for autonomous vehicle navigation,
    correcting drift in laser odometry using high-definition map data.

    Parameters
    ----------
    map_lanelet_path : str
        Path to the OpenStreetMap (.osm) file containing lanelet road network data.
    origin_coords_lanelet : list of float
        GPS origin coordinates [latitude, longitude] for map projection.
    angle_lanelet_correction : float
        Rotation angle in degrees to align map coordinate system with vehicle frame.
    frame_id : str, optional
        KITTI sequence identifier for logging and configuration, by default "00".

    Attributes
    ----------
    pose_history : list of Pose
        Sliding window of recent pose messages for trajectory alignment.
    lane_points : np.ndarray
        Processed lanelet centerline points in vehicle coordinate system.
    lane_kdtree : scipy.spatial.cKDTree
        Spatial index for efficient nearest neighbor queries on map points.

    Examples
    --------
    >>> # Initialize node with KITTI sequence 02 parameters
    >>> node = OdomCorrection(
    ...     map_lanelet_path="/path/to/lanelet2_seq_02.osm",
    ...     origin_coords_lanelet=[48.987607, 8.469747],
    ...     angle_lanelet_correction=-53.5,
    ...     frame_id="02"
    ... )
    >>> rclpy.spin(node)
    """

    def __init__(
        self, 
        map_lanelet_path: str = None
    ) -> None:
        super().__init__('osm_align_node')
        self.frame_count: int = 0
        # Declare ROS parameters with default values
        self.declare_parameter('frame_id', '00')
        self.declare_parameter('map_lanelet_path', '')
        self.declare_parameter('pose_history_size', 50)
        self.declare_parameter('knn_neighbors', 10)
        self.declare_parameter('valid_correspondence_threshold', 0.6)
        self.declare_parameter('icp_error_threshold', 1.5)
        self.declare_parameter('trimming_ratio', 0.2)
        self.declare_parameter('min_distance_threshold', 10.0)
        self.declare_parameter('odom_topic', '/liodom/odom')
        self.declare_parameter('transform_correction_service', '/liodom/transform_correction')
        

        
        # Get parameters
        self.frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value
        map_lanelet_path_param: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.pose_history_size: int = self.get_parameter('pose_history_size').get_parameter_value().integer_value
        self.knn_neighbors: int = self.get_parameter('knn_neighbors').get_parameter_value().integer_value
        self.valid_correspondence_threshold: float = self.get_parameter('valid_correspondence_threshold').get_parameter_value().double_value
        self.icp_error_threshold: float = self.get_parameter('icp_error_threshold').get_parameter_value().double_value
        self.trimming_ratio: float = self.get_parameter('trimming_ratio').get_parameter_value().double_value
        self.min_distance_threshold: float = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.transform_correction_service: str = self.get_parameter('transform_correction_service').get_parameter_value().string_value


        # Build map parameters from frame_id if not provided directly
        if map_lanelet_path is not None:
            self.map_lanelet_path: str = map_lanelet_path
        elif map_lanelet_path_param and map_lanelet_path_param.strip():
            self.map_lanelet_path: str = map_lanelet_path_param
        else:
            # Auto-construct path from frame_id
            self.map_lanelet_path: str = f'/home/joaquinecc/Documents/dataset/kitti/dataset/map/{self.frame_id}/lanelet2_seq_{self.frame_id}.osm'
            
        # Always use frame_id to get origin coordinates and angle correction from dictionaries
        self.origin_coords_lanelet: List[float] = [cordinta_dict[self.frame_id]['origin_lat'], cordinta_dict[self.frame_id]['origin_lon']]
        self.angle_lanelet_correction: float = angle_dict[self.frame_id]
                # Print all parameters for debugging/logging
        self.get_logger().info(
            f"Parameters:\n"
            f"  frame_id: {self.frame_id}\n"
            f"  map_lanelet_path: {map_lanelet_path_param}\n"
            f"  pose_history_size: {self.pose_history_size}\n"
            f"  knn_neighbors: {self.knn_neighbors}\n"
            f"  valid_correspondence_threshold: {self.valid_correspondence_threshold}\n"
            f"  icp_error_threshold: {self.icp_error_threshold}\n"
            f"  trimming_ratio: {self.trimming_ratio}\n"
            f"  min_distance_threshold: {self.min_distance_threshold}\n"
            f"  odom_topic: {self.odom_topic}\n"
            f"  transform_correction_service: {self.transform_correction_service}"
        )
        # Initialize lanelet-related variables
        self.projector: Optional[lanelet2.projection.UtmProjector] = None
        self.lanelet_map: Optional[lanelet2.core.LaneletMap] = None
        self.lane_kdtree: Optional[cKDTree] = None
        self.lane_cloud: Optional[np.ndarray] = None
        self.lane_points: Optional[np.ndarray] = None
        self.lane_points_next: Optional[np.ndarray] = None
        self.pose_history: List[Pose] = []
        
        # Load lanelet map and build KD-tree
        self._load_lanelet_map()
        self._build_lane_kdtree()
        
        # Create subscription
        self.subscription = self.create_subscription(   
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Create service client
        self.client = self.create_client(TransformCorrection, self.transform_correction_service)       
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.base_to_velo, _ = self.get_transform_matrix_from_tf(
            source_frame="base_link", 
            target_frame="velo_link", 
            timeout_sec=5
        )
        self.get_logger().info(f"Base to Velodyne transform matrix (from rclpy):\n{self.base_to_velo}")

    def _load_lanelet_map(self) -> None:
        """
        Load lanelet map from OpenStreetMap file using UTM projection.

        Initializes the UTM projector with the provided origin coordinates
        and loads the lanelet map data for subsequent processing.

        Raises
        ------
        Exception
            If the lanelet map file cannot be loaded or is corrupted.

        Notes
        -----
        The projector converts GPS coordinates to local UTM coordinates
        for efficient geometric computations during trajectory alignment.
        """
        self.projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(self.origin_coords_lanelet[0], self.origin_coords_lanelet[1])
        )
        self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self.projector)
        
        self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")
        self.get_logger().info(f"Origin coordinates: {self.origin_coords_lanelet}")
        self.get_logger().info(f"Angle correction: {self.angle_lanelet_correction} degrees")

    def _build_lane_kdtree(self) -> None:
        """
        Extract lanelet centerlines and build spatial index for efficient queries.

        Processes all lanelets in the map to extract centerline points, applies
        coordinate system rotation, removes duplicate points, and builds a KD-tree
        for fast nearest neighbor searches during trajectory alignment.

        The algorithm:
        1. Iterates through all lanelets and extracts centerline points
        2. Applies rotation matrix to align with vehicle coordinate system
        3. Filters points to maintain minimum spacing (1.0 meter)
        4. Creates "next point" associations for tangent vector computation
        5. Builds KD-tree spatial index

        Notes
        -----
        The minimum distance filtering (min_dist=1.0) reduces computational
        complexity while preserving map geometry for alignment purposes.
        """
        # Define rotation matrix
        rotation_angle = np.radians(self.angle_lanelet_correction)
        R_M = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        lane_points = []
        prev_point = None
        min_dist = 3.0
        lane_points_next = []
        
        for lanelet in self.lanelet_map.laneletLayer:
            centerline = lanelet.centerline
            aux_points = []
            for i, point in enumerate(centerline):
                corrected_point = R_M @ np.array([point.x, point.y])
                if prev_point is not None:
                    if np.linalg.norm(corrected_point - prev_point) < min_dist:
                        continue
                prev_point = corrected_point
                aux_points.append(corrected_point)
            lane_points.extend(aux_points)
            
            # Create next-point associations for tangent computation
            for i in range(len(aux_points)):
                nn = None
                if i < len(aux_points) - 1:
                    nn = aux_points[i + 1]
                else:
                    nn = aux_points[i - 1]
                lane_points_next.append(nn)

        self.lane_points_next = np.array(lane_points_next)
        self.lane_points = np.array(lane_points)
        self.lane_kdtree = cKDTree(self.lane_points)
        
        self.get_logger().info(f"Built KD-tree with {len(lane_points)} lane points")
    
    def _send_transform_correction(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> None:
        """
        Send 3D transform correction to the laser odometry service.

        Constructs a TransformCorrection service request containing a 
        geometry_msgs/Transform message with translation and rotation (as quaternion)
        to correct accumulated drift in the laser odometry system.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            Array of shape (3, 3) representing the rotation correction to apply
            in the world or map frame.
        translation_vector : np.ndarray
            Array of shape (3,) representing the translation correction to apply
            in meters [x, y, z].

        Examples
        --------
        >>> # Send a small correction
        >>> R = np.eye(3)  # No rotation correction
        >>> t = np.array([0.1, -0.05, 0.0])  # Small translation correction
        >>> node._send_transform_correction(R, t)

        Notes
        -----
        - The orientation difference is sent as a quaternion [x, y, z, w]
        - The service must be available and expect TransformCorrection message types
        - This is typically called after successful ICP alignment
        """
        req = TransformCorrection.Request()
        
        transform = Transform()
        transform.translation.x = float(translation_vector[0])
        transform.translation.y = float(translation_vector[1])
        transform.translation.z = float(translation_vector[2])
        
        quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        transform.rotation.x = float(quat[0])
        transform.rotation.y = float(quat[1])
        transform.rotation.z = float(quat[2])
        transform.rotation.w = float(quat[3])
        
        req.transform = transform
        self.client.call_async(req)

    def align_pose(self) -> None:
        """
        Align current pose history to lanelet map using robust ICP registration.

        This method implements the core trajectory-to-map alignment algorithm:
        1. Extracts (x, y) positions from the pose history sliding window
        2. Finds nearest lanelet map points using KD-tree spatial queries
        3. Computes optimal correspondences using normal shooting with tangent alignment
        4. Filters correspondences based on validity threshold
        5. Applies RANSAC-based ICP for robust transformation estimation
        6. Updates pose history and sends transform correction if successful

        The alignment process includes several robustness mechanisms:
        - Minimum trajectory distance requirement (10m) to ensure sufficient motion
        - Correspondence quality filtering to remove outliers
        - RANSAC-ICP with weighted sampling favoring recent poses
        - Error threshold validation before applying corrections

        Logs
        ----
        - Number of valid correspondences found
        - ICP final error and success/failure status  
        - Transformation parameters when alignment succeeds

        Notes
        -----
        Only x and y coordinates are transformed; z coordinates remain unchanged.
        The transformation is applied in-place to the pose history and also sent
        as a service request to the laser odometry system for real-time correction.
        """
        trajectory_points = [[pose.position.x, pose.position.y] for pose in self.pose_history]
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory_points, axis=0), axis=1))
        
        if total_distance < self.min_distance_threshold:
            self.get_logger().info(f"frame {self.frame_count} total_distance: {total_distance} < {self.min_distance_threshold}, skip ICP")
            return
            
        trajectory_points = np.array(trajectory_points)
        knn_index = self.lane_kdtree.query(trajectory_points, k=self.knn_neighbors)[1]
        best_lane_points = utils.find_interception_normal_shooting_nextpoint_tangent(
            trajectory_points, knn_index, self.lane_points, self.lane_points_next
        )
        valid_mask = ~np.isnan(best_lane_points).any(axis=1)

        if valid_mask.sum() < len(self.pose_history) * self.valid_correspondence_threshold:
            self.get_logger().info(f"frame {self.frame_count} ({valid_mask.sum()}) Not enough valid correspondences for ICP alignment")
            return

     
        R_total, T_total, final_error = utils.solve_trimmed_icp_2d(
            trajectory_points[valid_mask], 
            best_lane_points[valid_mask], 
            trimming_ratio=self.trimming_ratio,
        )
        if final_error < self.icp_error_threshold:
            self.get_logger().info(f"frame {self.frame_count} Pass threshold, ICP final error: {final_error}")

            pose_history = []
            self.pose_history = np.array(self.pose_history)
            
            # Apply 2D transformation to pose history
            for pose in self.pose_history:
                point_xy = np.array([pose.position.x, pose.position.y])
                point_xy = R_total @ point_xy + T_total
                pose.position.x = point_xy[0]
                pose.position.y = point_xy[1]  
                # pose.position.z remains unchanged
                pose_history.append(pose)
            self.pose_history = pose_history
            
            # Build 3D transformation matrix for service call
            R_3d = np.eye(3)
            R_3d[:2, :2] = R_total
            T_3d = np.zeros(3)
            T_3d[:2] = T_total
            
            self._send_transform_correction(R_3d, T_3d)
            self.get_logger().debug(f"frame {self.frame_count} T_total: {T_total}, R_total: {R_total}")
        else:
            self.get_logger().info(f"frame {self.frame_count} Fail threshold, ICP final error: {final_error}")
        
    def odom_callback(self, msg: Odometry) -> None:
        """
        Process incoming odometry messages and manage pose history sliding window.

        Transforms each incoming pose from base_link to velodyne frame, adds it
        to the pose history buffer, and triggers alignment when the buffer is full.
        Implements a sliding window approach to maintain computational efficiency.

        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Incoming odometry message containing pose and twist information.
            Only the pose component is used for alignment.

        Notes
        -----
        The sliding window approach:
        1. Accumulates poses until buffer reaches pose_history_size_
        2. Triggers alignment algorithm when buffer is full
        3. Removes oldest pose to maintain fixed buffer size
        4. Continues processing with updated poses

        This design balances alignment accuracy (sufficient trajectory length)
        with computational efficiency (bounded buffer size).
        """
        transformed_pose = utils.transform_pose(msg.pose.pose, self.base_to_velo)

        self.pose_history.append(transformed_pose)
        
        if len(self.pose_history) == self.pose_history_size:
            self.align_pose()
            self.pose_history.pop(0)
        self.frame_count += 1

    def get_transform_matrix_from_tf(
        self, 
        source_frame: str = "base_link", 
        target_frame: str = "velodyne", 
        timeout_sec: float = 2.0
    ) -> Tuple[np.ndarray, bool]:
        """
        Retrieve transformation matrix between coordinate frames using TF2.

        Queries the TF2 transform tree to obtain the homogeneous transformation
        matrix between two coordinate frames, typically used to convert poses
        from one reference frame to another (e.g., base_link to velodyne).

        Parameters
        ----------
        source_frame : str, default="base_link"
            Name of the source coordinate frame.
        target_frame : str, default="velodyne"  
            Name of the target coordinate frame.
        timeout_sec : float, default=2.0
            Maximum time to wait for the transform to become available.

        Returns
        -------
        transform_matrix : np.ndarray
            Homogeneous transformation matrix of shape (4, 4) that transforms
            points from source_frame to target_frame. Returns identity matrix
            if transform lookup fails.
        success : bool
            True if the transform was successfully retrieved, False otherwise.

        Examples
        --------
        >>> # Get base_link to velodyne transform
        >>> T, success = node.get_transform_matrix_from_tf("base_link", "velo_link")
        >>> if success:
        ...     print(f"Translation: {T[:3, 3]}")
        ...     print(f"Rotation matrix: {T[:3, :3]}")

        Notes
        -----
        The function converts ROS TransformStamped messages to homogeneous
        matrices for use in geometric computations. Handles quaternion to
        rotation matrix conversion using scipy's Rotation class.
        """
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,     # target frame (to)
                source_frame,     # source frame (from)
                rclpy.time.Time(seconds=0),  # latest available
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )    
            self.get_logger().info(f"Transform stamped: {transform_stamped}")
            
            # Extract translation
            translation = transform_stamped.transform.translation
            t = np.array([translation.x, translation.y, translation.z])
            
            # Extract rotation quaternion
            rotation = transform_stamped.transform.rotation
            quat = [rotation.x, rotation.y, rotation.z, rotation.w]
            
            # Convert quaternion to rotation matrix
            r = Rotation.from_quat(quat)
            R = r.as_matrix()
            
            # Create 4x4 homogeneous transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            self.get_logger().info(f"Successfully got transform from {source_frame} to {target_frame}")
            return transform_matrix, True
            
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform from {source_frame} to {target_frame}: {str(e)}")
            return np.eye(4), False


def main(args: Optional[List[str]] = None) -> None:
    """
    Initialize and run the OSM alignment node with ROS2 parameters.

    This function initializes ROS2, creates the OdomCorrection node, and spins
    it to process incoming messages. All configuration parameters are now
    handled through ROS2 parameters via the launch file or command line.

    Parameters
    ----------
    args : list of str, optional
        Command line arguments. If None, uses sys.argv.

    Examples
    --------
    >>> # Run with launch file (recommended)
    >>> # ros2 launch osm_align osm_align.launch.py frame_id:=02
    
    >>> # Run directly with ROS2 parameter syntax
    >>> # ros2 run osm_align kitti_odometry --ros-args -p frame_id:=02

    Notes
    -----
    Configuration parameters are now handled through ROS2 parameters:
    - Use the launch file for easy configuration with defaults
    - Parameters include frame_id, pose_history_size, ICP thresholds, etc.
    - The node automatically constructs map paths and configurations from frame_id
    """
    rclpy.init(args=args)
    
    # Create node - parameters will be read from ROS parameter system
    node = OdomCorrection()
    
    # Log the configuration being used
    node.get_logger().info(f"Starting OSM Alignment with frame_id: {node.frame_id}")
    node.get_logger().info(f"Map path: {node.map_lanelet_path}")
    node.get_logger().info(f"Origin coordinates: {node.origin_coords_lanelet}")
    node.get_logger().info(f"Angle correction: {node.angle_lanelet_correction}")
    node.get_logger().info(f"Pose history size: {node.pose_history_size}")
    node.get_logger().info(f"ICP error threshold: {node.icp_error_threshold}")
    node.get_logger().info(f"Odometry topic: {node.odom_topic}")
    node.get_logger().info(f"Transform service: {node.transform_correction_service}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
