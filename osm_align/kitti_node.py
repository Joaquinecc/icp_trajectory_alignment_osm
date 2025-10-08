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
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
import osm_align.utils.utils as utils
from scipy.linalg import inv
import time
from osm_align.odometry_correction import OdomCorrector
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu, NavSatFix

# Configuration parameters are now declared as ROS parameters in the node

BASE_FRAME="base_link"
SENSOR_FRAME="velo_link"
IMU_TOPIC="/kitti/oxts/imu"
GPS_TOPIC_NAME='/kitti/oxts/gps'
MIN_DIST_LANELET_POINTS=3.0 #3 meters
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
        self.map_lanelet_path: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.pose_segment_size: int = self.get_parameter('pose_segment_size').get_parameter_value().integer_value
        self.knn_neighbors: int = self.get_parameter('knn_neighbors').get_parameter_value().integer_value
        self.valid_correspondence_threshold: float = self.get_parameter('valid_correspondence_threshold').get_parameter_value().double_value
        self.icp_error_threshold: float = self.get_parameter('icp_error_threshold').get_parameter_value().double_value
        self.trimming_ratio: float = self.get_parameter('trimming_ratio').get_parameter_value().double_value
        self.min_distance_threshold: float = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.save_resuts_path: str = self.get_parameter('save_resuts_path').get_parameter_value().string_value

        self.broadcaster = StaticTransformBroadcaster(self)
        
        # Print all parameters for debugging/logging
        self.get_logger().info(
            f"Parameters:\n"
            f"  frame_id: {self.frame_id}\n"
            f"  map_lanelet_path: {self.map_lanelet_path}\n"
            f"  pose_segment_size: {self.pose_segment_size}\n"
            f"  knn_neighbors: {self.knn_neighbors}\n"
            f"  valid_correspondence_threshold: {self.valid_correspondence_threshold}\n"
            f"  icp_error_threshold: {self.icp_error_threshold}\n"
            f"  trimming_ratio: {self.trimming_ratio}\n"
            f"  min_distance_threshold: {self.min_distance_threshold}\n"
            f"  odom_topic: {self.odom_topic}\n"
            f"  save_resuts_path: {self.save_resuts_path}"
        )
        # This variables are initialized in the first gps_callback
        self.lanelet_map: Optional[lanelet2.core.LaneletMap] = None #Lanelet map
        self.lane_points: Optional[np.ndarray] = None #Lanelet points
        self.lane_points_nn: Optional[np.ndarray] = None #Lanelet points next-point associations
        self.origin_coords_lanelet: Optional[List[float]] = None #Origin gps coordinates of the lanelet map
        self.lane_kdtree: Optional[cKDTree] = None #Lanelet kdtree
        self.trajectory_correction: Optional[OdomCorrector] = None #Trajectory correction

        # This variables are initialized in the first imu_callback
        self.tf_odom_to_utm: Optional[np.ndarray] = None
        self.projector: Optional[lanelet2.projection.UtmProjector] = None #UTM projector

        # This variables are initialized in the first odom_callback
        self.pose_segment: List[Pose] = [] #Pose history
        self.delta_t_acc=np.eye(4) #Delta t accumulator
        self.poses_history: List[np.ndarray] = [] #Pose history
        self.align_runtimes: List[float] = [] #Alignment runtime
        
        self.correction_args={
            'pose_segment_size': self.pose_segment_size,
            'knn_neighbors': self.knn_neighbors,
            'valid_correspondence_threshold': self.valid_correspondence_threshold,
            'icp_error_threshold': self.icp_error_threshold,
            'trimming_ratio': self.trimming_ratio,
            'min_distance_threshold': self.min_distance_threshold,
        }


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_base_to_imu, _ = self.get_transform_matrix_from_tf(
            source_frame="base_link", 
            target_frame="imu_link", 
            timeout_sec=5
        )

        # Create subscription
        self.subscription = self.create_subscription(   
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        self.publisher_odom=self.create_publisher(Odometry, '/osm_align/odom_aligned', 10)
        
        self.subscription_imu = self.create_subscription(
            Imu,
            IMU_TOPIC,
            self.imu_callback,
            10
        )
        self.subscription_gps = self.create_subscription(
            NavSatFix,
            GPS_TOPIC_NAME,
            self.gps_callback,
            10
        )

        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_pub = self.create_publisher(MarkerArray, '/osm_align/lanelet_markers', qos)



    def imu_callback(self, msg: Imu) -> None:
        """
        Callback processing incoming IMU messages.
        """
        #To get initial orientation of the vehicle
        if self.tf_odom_to_utm is None:
            self.get_logger().info(f"Initial orientation of the vehicle in ENU coordinates: {msg}")
            #Get orientation of the vehicle in ENU coordinates
            quat_ori= np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            quat_ori= Rotation.from_quat(quat_ori).as_matrix()
            #Convert to UTM coordinates
            self.tf_odom_to_utm=np.identity(4)
            self.tf_odom_to_utm[:3,:3] = quat_ori[:3,:3]@self.tf_base_to_imu[:3,:3]
            self.get_logger().info(f"TF to UTM matrix: {self.tf_odom_to_utm}")
            self.publish_transform_to_utm()

    def publish_transform_to_utm(self) -> None:
        """
        Publish the transform to the IMU frame.
        """
        transform_stamped = TransformStamped()
        transform_stamped.header.frame_id = "utm"
        transform_stamped.child_frame_id = "odom"
        transform_stamped.transform.translation.x = self.tf_odom_to_utm[0, 3]
        transform_stamped.transform.translation.y = self.tf_odom_to_utm[1, 3]
        transform_stamped.transform.translation.z = self.tf_odom_to_utm[2, 3]
        quat = Rotation.from_matrix(self.tf_odom_to_utm[:3, :3]).as_quat()
        transform_stamped.transform.rotation.x = quat[0]
        transform_stamped.transform.rotation.y = quat[1]
        transform_stamped.transform.rotation.z = quat[2]
        transform_stamped.transform.rotation.w = quat[3]
        self.broadcaster.sendTransform(transform_stamped)

        self.get_logger().info(f"Published transform to UTM frame: {self.tf_odom_to_utm}")
    def gps_callback(self, msg: NavSatFix) -> None:
        """
        Callback processing incoming GPS messages.
        """
        if self.origin_coords_lanelet is None:
            self.get_logger().info(f"Initial position of the vehicle in ENU coordinates: {msg}")
            #Get position of the vehicle in ENU coordinates
            self.origin_coords_lanelet= np.array([msg.latitude, msg.longitude, msg.altitude])
            self.projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(self.origin_coords_lanelet[0], self.origin_coords_lanelet[1])
            )
            #Load lanelet map   
            self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self.projector)
            self.get_logger().info(f"Origin coordinates: {self.origin_coords_lanelet}")
            #Build lanelet points and its next-point associations
            lane_points, lane_points_nn = utils.lane_points_and_it_nn(self.lanelet_map, MIN_DIST_LANELET_POINTS)
            #Build KD-tree
            self.lane_points_nn = np.array(lane_points_nn)
            self.lane_points = np.array(lane_points)
            self.lane_kdtree = cKDTree(self.lane_points)

            self.trajectory_correction=OdomCorrector(self.lane_points, self.lane_points_nn, self.lane_kdtree, self.correction_args)
            #Publish lanelet markers
            self.publish_lanelet_markers()



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
        odom_msg.child_frame_id = "base_link"
        

        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose=pose
        self.publisher_odom.publish(odom_msg)

        self.get_logger().debug(f"frame {self.frame_count} pose: {pose.position.x}, {pose.position.y}, {pose.position.z}")

    def publish_lanelet_markers(self) -> None:
        """
        Publish the lanelet markers for RVIZ visualization.
        """
        #Map settings for RVIZ visualization
        markers = MarkerArray()
        for idx, lanelet in enumerate(self.lanelet_map.laneletLayer):
            line=[[pt.x, pt.y] for pt in lanelet.centerline]
            line=np.array(line)
            marker = Marker()
            marker.header.frame_id = "utm"
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            # marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'lanelet_centerlines'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(0.2)
            marker.color.r = float(188.0 / 255.0)
            marker.color.g = float(203.0 / 255.0)
            marker.color.b = float(169.0 / 255.0)
            marker.color.a = float(1.0)
            # Convert to geometry_msgs/Point list, z=0
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in line]
            # Infinite lifetime; with transient local, late subscribers will receive
            marker.lifetime = Duration(seconds=0).to_msg()
            markers.markers.append(marker)

        self.map_pub.publish(markers)



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
        if self.origin_coords_lanelet is None:
            return
        #move to velodyne frame
        tf= self.tf_odom_to_utm if self.tf_odom_to_utm is not None else np.identity(4)
        transformed_pose = tf@utils.pose_to_4x4(msg.pose.pose) #Transform to UTM frame
        self.get_logger().info(f"origina {utils.pose_to_4x4(msg.pose.pose) } transformed_pose: {transformed_pose} ")
        t0 = time.perf_counter()
        pose_corrected, message=self.trajectory_correction.apply(transformed_pose)
        dt = time.perf_counter() - t0

        # self.get_logger().info(f"frame {self.frame_count} align runtime: {dt}")

        if message==0:
            self.get_logger().info(f"frame {self.frame_count} trajectory length < {self.min_distance_threshold}, skip ICP")
        elif message==1:
            self.get_logger().info(f"frame {self.frame_count} ICP error > {self.icp_error_threshold}, skip ICP")
        elif message==-1:
            pass
        else:
            self.get_logger().info(f"frame {self.frame_count} ICP error < {message}, ICP success")

        self.align_runtimes.append(dt)
        self.frame_count += 1


        # Record pose to history before publishing
        pose_corrected=inv(self.tf_odom_to_utm)@pose_corrected
        self.poses_history.append(pose_corrected)

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
            target_frame: str = "velo_link", 
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
            matrices for use in geomesstric computations. Handles quaternion to
            rotation matrix conversion using scipy's Rotation class.
            """
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    target_frame,     # target frame (to)
                    source_frame,     # source frame (from)
                    rclpy.time.Time(seconds=0),  # latest available
                    timeout=rclpy.duration.Duration(seconds=timeout_sec)
                )    
                self.get_logger().debug(f"Transform stamped: {transform_stamped}")
                
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
                raise e


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
