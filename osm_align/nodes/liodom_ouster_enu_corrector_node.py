# Copyright 2025 Distance Technologies Oy. For internal use only.
#
"""
Node for correcting liodom odometry to ENU format with proper yaw alignment.
It subscribes to liodom odometry, initial GPS (for origin), and GPS data.
It applies a 180-degree yaw rotation (for Ouster sensor) and calculates the ENU yaw offset
using consecutive odometry and GPS measurements.
"""

# ROS2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
import tf2_ros

# Lanelet2
import lanelet2
from lanelet2.core import GPSPoint

# Python libraries
from scipy.spatial.transform import Rotation
import numpy as np
from typing import Optional

# Local libraries
from osm_align.utils import utils

ODOM_ENU_TOPIC = "/liodom/odom_enu"


class LiodomOusterEnuCorrectorNode(Node):
    """
    Node that corrects liodom odometry to ENU format with proper yaw alignment.
    """

    def __init__(self):
        super().__init__("liodom_ouster_enu_corrector_node")
        self.get_logger().info("Liodom Ouster ENU corrector node initialized")
        
        # Declare parameters
        self.declare_parameter('odom_topic', '/liodom/odom')
        self.declare_parameter('initial_gps_topic', '/Inertial_Labs/initial_gps')
        self.declare_parameter('gps_topic', '/Inertial_Labs/gps_data_std')
        
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.initial_gps_topic: str = self.get_parameter('initial_gps_topic').get_parameter_value().string_value
        self.gps_topic: str = self.get_parameter('gps_topic').get_parameter_value().string_value
        
        self.get_logger().info(f"Parameters:\n"
                                f"  odom_topic: {self.odom_topic}\n"
                                f"  initial_gps_topic: {self.initial_gps_topic}\n"
                                f"  gps_topic: {self.gps_topic}")

        # Initialize variables
        self._utm_projector: Optional[lanelet2.projection.UtmProjector] = None
        self._utm_origin: Optional[tuple] = None
        self.poses_history: list = []  # List of 4x4 pose matrices (in map frame after rotation)
        self.enu_yaw_offset: Optional[float] = None
        self.tf_to_yaw_enu_correction: np.ndarray = np.eye(4)  # Correction transform
        self.initial_gps_received: bool = False
        self.gps_frame_id: Optional[str] = None
        
        # 180-degree rotation matrix for Ouster sensor (rotation around Z-axis, orientation only)
        self.ouster_180_rotation_matrix: np.ndarray = Rotation.from_euler('z', [180], degrees=True).as_matrix()
        
        # QOS Settings for initial GPS (transient local for late subscribers)
        initial_gps_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        
        # Subscribers
        self.sub_initial_gps = self.create_subscription(
            NavSatFix,
            self.initial_gps_topic,
            self.initial_gps_callback,
            initial_gps_qos
        )
        
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        self.sub_gps = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            10
        )
        
        # Publisher
        self.pub_odom_enu = self.create_publisher(Odometry, ODOM_ENU_TOPIC, 10)
        
        # TF buffer for coordinate frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        
     

    def initial_gps_callback(self, msg: NavSatFix) -> None:
        """
        Handle initial GPS message to set the origin.
        This should be called once to establish the origin pose.
        """
        if self.initial_gps_received:
            return  # Only process once
        
        lat0 = float(msg.latitude)
        lon0 = float(msg.longitude)
        alt0 = float(msg.altitude)
        
        # Initialize UTM projector with origin
        self._utm_origin = (lat0, lon0)
        self._utm_projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(lat0, lon0)
        )
        self.gps_frame_id = msg.header.frame_id
        
        self.get_logger().info(
            f"Initial GPS received - Origin set: lat={lat0:.8f}, lon={lon0:.8f}, alt={alt0:.2f}"
        )
        
        # The origin pose will be set when we receive the first odometry message
        # at the same time as the initial GPS
        self.initial_gps_received = True

    def odom_callback(self, msg: Odometry) -> None:
        """
        Handle liodom odometry messages.
        Transforms from odom to map frame, applies 180-degree rotation for Ouster sensor,
        then ENU yaw correction if available.
        """
        if not self._utm_projector:
            self.get_logger().warn("UTM projector not initialized yet, waiting for initial GPS...")
            
        
        # Convert pose to 4x4 matrix (in odom frame)
        pose_corrected = utils.pose_to_4x4(msg.pose.pose)
        # Apply ENU yaw correction if available
        if self.enu_yaw_offset is not None:
            pose_corrected = self.tf_to_yaw_enu_correction @ pose_corrected
        else:
            pose_corrected = pose_corrected
       
        pose_corrected[:3, :3] = self.ouster_180_rotation_matrix @ pose_corrected[:3, :3]
        self.poses_history.append(pose_corrected)
        
        # Publish corrected odometry
        self.publish_odom_enu(pose_corrected, msg)

    def gps_callback(self, msg: NavSatFix) -> None:
        """
        Handle consecutive GPS messages to calculate ENU yaw offset.
        Uses the difference between GPS position and odometry position to estimate yaw offset.
        """
        if not self._utm_projector:
            return
        
        if len(self.poses_history) < 1:
            return  # Need at least 1 pose to calculate yaw offset
        
        lat0 = float(msg.latitude)
        lon0 = float(msg.longitude)
        alt0 = float(msg.altitude)
        
        # Convert GPS to UTM coordinates (relative to UTM origin)
        ref_point = self._utm_projector.forward(GPSPoint(lat0, lon0, alt0))
        ref_point_xy = [ref_point.x, ref_point.y]
        
        # Get the latest pose position (after 180-degree rotation, in map frame)
        target_point = self.poses_history[-1][:2, -1]
        
        # Calculate yaw offset between GPS direction and odometry direction
        yaw_offset = utils.rotation_angle_2d(ref_point_xy, target_point)
        
        # Update yaw offset and correction transform
        self.enu_yaw_offset = yaw_offset
        self.tf_to_yaw_enu_correction[:3, :3] = Rotation.from_euler(
            'z', [-self.enu_yaw_offset], degrees=True
        ).as_matrix()
        
        self.get_logger().info(f"ENU yaw offset calculated: {self.enu_yaw_offset:.2f} degrees")
        self.destroy_subscription(self.sub_gps)

    def publish_odom_enu(self, pose_4x4: np.ndarray, original_msg: Odometry) -> None:
        """
        Publish corrected odometry in ENU format.
        
        Parameters
        ----------
        pose_4x4 : np.ndarray
            4x4 homogeneous transformation matrix
        original_msg : Odometry
            Original odometry message to copy metadata from
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = original_msg.header.stamp
        odom_msg.header.frame_id = "odom"  # ENU frame
        odom_msg.child_frame_id = original_msg.child_frame_id
        
        # Extract position
        odom_msg.pose.pose.position.x = float(pose_4x4[0, 3])
        odom_msg.pose.pose.position.y = float(pose_4x4[1, 3])
        odom_msg.pose.pose.position.z = float(pose_4x4[2, 3])
        
        # Extract orientation (convert rotation matrix to quaternion)
        quat_xyzw = Rotation.from_matrix(pose_4x4[:3, :3]).as_quat()
        odom_msg.pose.pose.orientation.x = float(quat_xyzw[0])
        odom_msg.pose.pose.orientation.y = float(quat_xyzw[1])
        odom_msg.pose.pose.orientation.z = float(quat_xyzw[2])
        odom_msg.pose.pose.orientation.w = float(quat_xyzw[3])
        
        # Copy twist if available (could also apply rotation if needed)
        odom_msg.twist = original_msg.twist
        
        self.pub_odom_enu.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LiodomOusterEnuCorrectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

