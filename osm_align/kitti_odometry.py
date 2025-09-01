"""ROS2 node for odometry alignment against Lanelet2 centerlines using trimmed ICP."""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import numpy as np
from typing import List, Tuple, Optional
import os
import lanelet2
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from tf2_ros import Buffer, TransformListener
import osm_align.utils.utils as utils
from osm_align.utils.kitti_utils import angle_dict, cordinta_dict
from scipy.linalg import inv
import time
from osm_align.odometry_correction import OdomCorrector
# Configuration parameters are now declared as ROS parameters in the node


class KittiOdometryCorrection(Node):
    """
    ROS2 node that aligns odometry to OSM/Lanelet2 centerlines.

    The node subscribes to an odometry topic, transforms poses into the
    sensor frame, applies a dataset-agnostic odometry correction via
    `OdomCorrector`, and republishes the corrected odometry. It also
    supports saving alignment runtime and trajectory matrices.

    Parameters
    ----------
    map_lanelet_path : Optional[str], default None
        Optional external path to a Lanelet2 map. If not provided, a path is
        constructed automatically from `frame_id`.
    """


    def __init__(
        self, 
        map_lanelet_path: Optional[str] = None
    ) -> None:
        super().__init__('kitti_odometry_correction_node')
        self.frame_count: int = 0
        # Declare ROS parameters with default values
        self.declare_parameter('frame_id', '00')
        self.declare_parameter('map_lanelet_path', '')
        self.declare_parameter('pose_segment_size', 50)
        self.declare_parameter('knn_neighbors', 10)
        self.declare_parameter('valid_correspondence_threshold', 0.6)
        self.declare_parameter('icp_error_threshold', 1.5)
        self.declare_parameter('trimming_ratio', 0.2)
        self.declare_parameter('min_distance_threshold', 10.0)
        self.declare_parameter('odom_topic', '/liodom/odom')
        self.declare_parameter('save_resuts_path', '')
        

        
        # Get parameters
        self.frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value
        map_lanelet_path_param: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.pose_segment_size: int = self.get_parameter('pose_segment_size').get_parameter_value().integer_value
        self.knn_neighbors: int = self.get_parameter('knn_neighbors').get_parameter_value().integer_value
        self.valid_correspondence_threshold: float = self.get_parameter('valid_correspondence_threshold').get_parameter_value().double_value
        self.icp_error_threshold: float = self.get_parameter('icp_error_threshold').get_parameter_value().double_value
        self.trimming_ratio: float = self.get_parameter('trimming_ratio').get_parameter_value().double_value
        self.min_distance_threshold: float = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.save_resuts_path: str = self.get_parameter('save_resuts_path').get_parameter_value().string_value
        self.map_lanelet_path: str = f'/home/joaquinecc/Documents/dataset/kitti/dataset/map/{self.frame_id}/lanelet2_seq_{self.frame_id}.osm'
            
        # Always use frame_id to get origin coordinates and angle correction from dictionaries
        self.origin_coords_lanelet: List[float] = [cordinta_dict[self.frame_id]['origin_lat'], cordinta_dict[self.frame_id]['origin_lon']]
        self.angle_lanelet_correction: float = angle_dict[self.frame_id]
                # Print all parameters for debugging/logging
        self.get_logger().info(
            f"Parameters:\n"
            f"  frame_id: {self.frame_id}\n"
            f"  map_lanelet_path: {map_lanelet_path_param}\n"
            f"  pose_segment_size: {self.pose_segment_size}\n"
            f"  knn_neighbors: {self.knn_neighbors}\n"
            f"  valid_correspondence_threshold: {self.valid_correspondence_threshold}\n"
            f"  icp_error_threshold: {self.icp_error_threshold}\n"
            f"  trimming_ratio: {self.trimming_ratio}\n"
            f"  min_distance_threshold: {self.min_distance_threshold}\n"
            f"  odom_topic: {self.odom_topic}\n"
            f"  save_resuts_path: {self.save_resuts_path}"
        )
        # Initialize lanelet-related variables
        self.projector: Optional[lanelet2.projection.UtmProjector] = None
        self.lanelet_map: Optional[lanelet2.core.LaneletMap] = None
        self.lane_points: Optional[np.ndarray] = None
        self.lane_points_next: Optional[np.ndarray] = None
        self.pose_segment: List[Pose] = []
        self.delta_t_acc=np.eye(4)
        # New: history buffers
        self.poses_history: List[np.ndarray] = []
        self.align_runtimes: List[float] = []
        # Load lanelet map and build KD-tree
        self._load_lanelet_map()
        self._build_lane_kdtree()
        



        args={
            'pose_segment_size': self.pose_segment_size,
            'knn_neighbors': self.knn_neighbors,
            'valid_correspondence_threshold': self.valid_correspondence_threshold,
            'icp_error_threshold': self.icp_error_threshold,
            'trimming_ratio': self.trimming_ratio,
            'min_distance_threshold': self.min_distance_threshold,
        }
        self.trajectory_correction=OdomCorrector(self.lane_points, self.lane_points_next, self.lane_kdtree, args)


        # Create subscription
        self.subscription = self.create_subscription(   
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        self.publisher_odom=self.create_publisher(Odometry, '/osm_align/odom_aligned', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)


        self.base_to_velo, _ = self.get_transform_matrix_from_tf(
            source_frame="base_link", 
            target_frame="velo_link", 
            timeout_sec=5
        )

        self.get_logger().debug(f"Base to Velodyne transform matrix (from rclpy):\n{self.base_to_velo}")

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
        min_dist = 3.0
        lane_points_next = []
        
        for lanelet in self.lanelet_map.laneletLayer:
            prev_point = None
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
 
    def publish_odom(self, pose: Pose) -> None:
        """
        Publish the aligned odometry message.
        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            Pose to publish.

        Notes
        -----
        The pose is published as an Odometry message with the pose in the pose field.   
        """
        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom" 
        odom_msg.child_frame_id = "velo_link"
        

        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose=pose
        self.publisher_odom.publish(odom_msg)

        self.get_logger().debug(f"frame {self.frame_count} pose: {pose.position.x}, {pose.position.y}, {pose.position.z}")

    def _pose_to_4x4(self, pose: Pose) -> np.ndarray:
        """
        Convert geometry_msgs/Pose to a 4x4 homogeneous transformation matrix.

        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            Input pose with position and orientation (quaternion).

        Returns
        -------
        numpy.ndarray
            A 4x4 homogeneous matrix in row-major layout.
        """
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        R3 = Rotation.from_quat(quat).as_matrix()
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        M = np.eye(4)
        M[:3, :3] = R3
        M[:3, 3] = t
        return M

    def save_results(self) -> None:
        """Save pose history and alignment runtimes if path is provided."""
        if not self.save_resuts_path or not self.save_resuts_path.strip():
            return
        try:
            self.save_resuts_path=os.path.join(self.save_resuts_path, self.frame_id)
            os.makedirs(self.save_resuts_path, exist_ok=True)
            poses_path = os.path.join(self.save_resuts_path, 'poses.txt')
            with open(poses_path, 'w') as f:
                for M in self.poses_history:
                    vals = M.reshape(-1)
                    f.write(' '.join(f'{v:.12f}' for v in vals) + '\n')
            runtime_path = os.path.join(self.save_resuts_path, 'runtime.txt')
            with open(runtime_path, 'w') as f:
                for rt in self.align_runtimes:
                    f.write(f'{rt:.6f}\n')
            self.get_logger().info(f"Saved results to: {self.save_resuts_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to save results: {e}")

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback processing incoming odometry messages.

        The pose is transformed into the sensor frame, corrected using
        `OdomCorrector`, appended to history, and then transformed back
        before republishing on the aligned odometry topic.

        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Incoming odometry message.
        """
        #move to velodyne frame
        transformed_pose = self.base_to_velo@self._pose_to_4x4(msg.pose.pose)
        t0 = time.perf_counter()
        pose_corrected=self.trajectory_correction.apply(transformed_pose)
        dt = time.perf_counter() - t0

        self.get_logger().info(f"frame {self.frame_count} align runtime: {dt}")
        self.align_runtimes.append(dt)
        self.frame_count += 1
        self.poses_history.append(pose_corrected)


        # Record pose to history before publishing
        pose_corrected = inv(self.base_to_velo) @pose_corrected
        pose_recived=msg.pose.pose
        pose_recived.position.x=pose_corrected[0, -1]
        pose_recived.position.y=pose_corrected[1, -1]
        pose_recived.position.z=pose_corrected[2, -1]
        pose_recived.orientation.x=pose_corrected[0, 0]
        pose_recived.orientation.y=pose_corrected[1, 0]
        pose_recived.orientation.z=pose_corrected[2, 0]
        pose_recived.orientation.w=pose_corrected[3, 0]
        self.publish_odom(pose_recived)


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
    - Parameters include frame_id, pose_segment_size, ICP thresholds, etc.
    - The node automatically constructs map paths and configurations from frame_id
    """
    rclpy.init(args=args)
    
    # Create node - parameters will be read from ROS parameter system
    node = KittiOdometryCorrection()
    
    # Log the configuration being used
    node.get_logger().info(f"Starting OSM Alignment with frame_id: {node.frame_id}")
    node.get_logger().info(f"Map path: {node.map_lanelet_path}")
    node.get_logger().info(f"Origin coordinates: {node.origin_coords_lanelet}")
    node.get_logger().info(f"Angle correction: {node.angle_lanelet_correction}")
    node.get_logger().info(f"Pose history size: {node.pose_segment_size}")
    node.get_logger().info(f"ICP error threshold: {node.icp_error_threshold}")
    node.get_logger().info(f"Odometry topic: {node.odom_topic}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        # Save results if requested
        try:
            node.save_results()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
