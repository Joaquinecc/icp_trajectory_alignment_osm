# Copyright 2026 Distance Technologies Oy. For internal use only.
#
"""
Generic node for calculating yaw offset and publishing corrected odometry.
It subscribes to odometry and GPS topics, calculates the yaw offset based on
GPS position and odometry trajectory, then publishes corrected odometry to a new topic.
"""

# ROS2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

# Lanelet2
import lanelet2
from lanelet2.core import GPSPoint

# Python libraries
from scipy.spatial.transform import Rotation
import numpy as np
from typing import Optional

# Local libraries
from osm_align.utils import utils


class OdomEnuCorrectionNode(Node):
    """
    ROS2 node that calculates yaw offset between odometry and GPS trajectories.
    
    This node subscribes to odometry and GPS topics, collects a set of corresponding
    poses, and uses the Kabsch algorithm to compute the optimal yaw rotation that
    aligns the odometry trajectory with the GPS trajectory. Once calculated, it
    applies this yaw correction to all subsequent odometry messages and publishes
    the corrected odometry.
    
    Attributes
    ----------
    _utm_projector : Optional[lanelet2.projection.UtmProjector]
        UTM coordinate projector initialized from first GPS message.
    _utm_origin : Optional[tuple]
        GPS origin coordinates (latitude, longitude) used for UTM projection.
    poses_history : list
        List of 4x4 pose matrices from odometry messages (up to n_points).
    gps_positions : list
        List of GPS positions in ENU coordinates relative to origin (up to n_points).
    odom_origin : Optional[np.ndarray]
        First odometry pose used as reference.
    enu_origin : Optional[np.ndarray]
        First GPS position in ENU coordinates.
    yaw_offset_deg : Optional[float]
        Calculated yaw offset in degrees, None until computed.
    n_points : int
        Number of points to collect before calculating yaw offset.
    """

    def __init__(self):
        """
        Initialize the OdomEnuCorrectionNode.
        
        Declares ROS2 parameters for input/output topics and initializes
        subscribers, publishers, and internal state variables.
        """
        super().__init__("odom_enu_correction_node")
        self.get_logger().info("Odom ENU correction node initialized")
        
        # Declare parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('gps_topic', '/gps')
        self.declare_parameter('odom_output_topic', '')
        
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.gps_topic: str = self.get_parameter('gps_topic').get_parameter_value().string_value
        odom_output_topic_param: str = self.get_parameter('odom_output_topic').get_parameter_value().string_value
        
        # Use provided output topic or default to f"{odom_topic}_enu"
        if odom_output_topic_param and odom_output_topic_param.strip():
            self.odom_enu_topic = odom_output_topic_param
        else:
            self.odom_enu_topic = f"{self.odom_topic}_enu"
        
        self.get_logger().info(f"Parameters:\n"
                                f"  odom_topic: {self.odom_topic}\n"
                                f"  gps_topic: {self.gps_topic}\n"
                                f"  odom_output_topic: {self.odom_enu_topic}")

        # Initialize variables
        self._utm_projector: Optional[lanelet2.projection.UtmProjector] = None
        self._utm_origin: Optional[tuple] = None
        self.poses_history: list = []  # List of 4x4 pose matrices (in odom frame) - max n
        self.gps_positions: list = []  # List of GPS positions in ENU (relative to origin) - max n
        self.odom_origin: Optional[np.ndarray] = None  # First odom pose
        self.enu_origin: Optional[np.ndarray] = None  # First GPS position in ENU
        self.yaw_offset_deg: Optional[float] = None  # Calculated yaw offset in degrees
        self.n_points: int = 10  # Number of points to collect for Kabsch algorithm
        
        # Subscribers
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
        
        # Publisher for corrected odometry
        self.pub_odom_enu = self.create_publisher(
            Odometry,
            self.odom_enu_topic,
            10
        )
        self.get_logger().info(f"Publishing corrected odometry to: {self.odom_enu_topic}")

    def odom_callback(self, msg: Odometry) -> None:
        """
        Handle odometry messages.
        
        Stores poses for yaw offset calculation and publishes corrected odometry.
        If yaw offset has been calculated, applies the correction before publishing.
        
        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Incoming odometry message.
        """
        # Convert pose to 4x4 matrix (in odom frame)
        pose_4x4 = utils.pose_to_4x4(msg.pose.pose)
        
        # Store first odom pose as origin if not set
        if self.odom_origin is None:
            self.odom_origin = pose_4x4.copy()
        
        # Store pose history (up to n_points)
        if len(self.poses_history) < self.n_points:
            self.poses_history.append(pose_4x4)
        
        # Try to calculate yaw offset if we have enough poses and GPS positions
        if (len(self.poses_history) >= self.n_points and 
            len(self.gps_positions) >= self.n_points and 
            self.yaw_offset_deg is None):
            self._try_calculate_yaw_offset()
        
        # If yaw offset is calculated, apply it and publish corrected odometry
        if self.yaw_offset_deg is not None:
            corrected_msg = self._apply_yaw_correction(msg)
        else:
            corrected_msg = msg
        self.pub_odom_enu.publish(corrected_msg)

    def gps_callback(self, msg: NavSatFix) -> None: 
        """
        Handle GPS messages to calculate ENU transform.
        
        First GPS message initializes the UTM projector and sets the origin.
        Subsequent messages are converted to ENU coordinates and stored for
        yaw offset calculation.
        
        Parameters
        ----------
        msg : sensor_msgs.msg.NavSatFix
            Incoming GPS message.
        """
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        alt = float(msg.altitude)
        
        # Initialize UTM projector with first GPS message as origin
        if self._utm_projector is None:
            self._utm_origin = (lat, lon)
            self._utm_projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(lat, lon)
            )
            
            # First GPS position will be at origin (0, 0, alt)
            self.enu_origin = np.array([0.0, 0.0, float(alt)])
            self.gps_positions.append(self.enu_origin.copy())
            
            self.get_logger().info(
                f"GPS origin set: lat={lat:.8f}, lon={lon:.8f}, alt={alt:.2f}, "
                f"ENU=({self.enu_origin[0]:.2f}, {self.enu_origin[1]:.2f}, {self.enu_origin[2]:.2f})"
            )
        else:
            # Convert GPS to UTM/ENU coordinates (relative to origin)
            ref_point = self._utm_projector.forward(GPSPoint(lat, lon, alt))
            gps_pos = np.array([float(ref_point.x), float(ref_point.y), float(ref_point.z)])
            # Store GPS positions (up to n_points)
            if len(self.gps_positions) < self.n_points:
                self.gps_positions.append(gps_pos)
        
        # Try to calculate yaw offset if we have enough poses and GPS positions
        if (len(self.poses_history) >= self.n_points and 
            len(self.gps_positions) >= self.n_points and 
            self.yaw_offset_deg is None):
            self._try_calculate_yaw_offset()

    def _try_calculate_yaw_offset(self) -> None:
        """
        Calculate yaw offset from odom to ENU using Kabsch algorithm on 2D points.
        
        Extracts 2D positions from collected odometry poses and GPS positions,
        then uses the Kabsch algorithm to compute the optimal rotation matrix.
        The yaw angle is extracted from this rotation matrix and stored.
        Unsubscribes from GPS topic after successful calculation.
        
        Requires at least n_points GPS positions and odometry poses to be collected.
        """
        if len(self.poses_history) < self.n_points or len(self.gps_positions) < self.n_points:
            return
        
        # Extract 2D positions (x, y) from odom poses - ignore z/altitude
        odom_points_2d = np.array([pose[:2, 3] for pose in self.poses_history[:self.n_points]])
   
        
        # Extract 2D positions (x, y) from GPS - ignore z/altitude
        gps_points_2d = np.array([gps_pos[:2] for gps_pos in self.gps_positions[:self.n_points]])
        
        # Use Kabsch algorithm to find optimal rotation
        # Align odom points (source) to GPS points (target)
        rotation_matrix = utils.kabsch_2d(odom_points_2d, gps_points_2d)
        
        if rotation_matrix is None:
            self.get_logger().warn("Failed to calculate rotation using Kabsch algorithm")
            return
        
        # Extract yaw angle from 2D rotation matrix
        # For 2D rotation: R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        # yaw = atan2(R[1,0], R[0,0]) or atan2(-R[0,1], R[0,0])
        yaw_offset_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        self.yaw_offset_deg = np.degrees(yaw_offset_rad)
        
        self.get_logger().info(
            f"Calculated yaw offset using Kabsch algorithm: {self.yaw_offset_deg:.2f} degrees "
            f"(from {self.n_points} poses and {self.n_points} GPS positions)"
        )
        
        # Unsubscribe from GPS after calculation
        self.destroy_subscription(self.sub_gps)
    

    
    def _apply_yaw_correction(self, msg: Odometry) -> Odometry:
        """
        Apply yaw correction to odometry message.
        
        Parameters
        ----------
        msg : Odometry
            Original odometry message
            
        Returns
        -------
        Odometry
            Corrected odometry message with yaw offset applied
        """
        # Convert pose to 4x4 matrix
        pose_4x4 = utils.pose_to_4x4(msg.pose.pose)
        
        # Create yaw correction rotation matrix
        yaw_correction=np.eye(4)
        yaw_correction[:3, :3] = Rotation.from_euler(
            'z', [-self.yaw_offset_deg], degrees=True
        ).as_matrix()
        
        # Apply yaw correction to rotation
        corrected_pose_4x4 = yaw_correction @ pose_4x4
        
        # Create new odometry message
        corrected_msg = Odometry()
        corrected_msg.header = msg.header
        corrected_msg.header.frame_id = msg.header.frame_id
        corrected_msg.child_frame_id = msg.child_frame_id
        
        # Copy position (now with yaw correction applied)
        corrected_msg.pose.pose.position.x = float(corrected_pose_4x4[0, 3])
        corrected_msg.pose.pose.position.y = float(corrected_pose_4x4[1, 3])
        corrected_msg.pose.pose.position.z = float(corrected_pose_4x4[2, 3])
        
        # Convert corrected rotation to quaternion
        quat_xyzw = Rotation.from_matrix(corrected_pose_4x4[:3, :3]).as_quat()
        corrected_msg.pose.pose.orientation.x = float(quat_xyzw[0])
        corrected_msg.pose.pose.orientation.y = float(quat_xyzw[1])
        corrected_msg.pose.pose.orientation.z = float(quat_xyzw[2])
        corrected_msg.pose.pose.orientation.w = float(quat_xyzw[3])
        
        # Copy covariance
        corrected_msg.pose.covariance = msg.pose.covariance
        
        # Apply yaw correction to twist (angular velocity around z-axis)
        # Angular velocity is a vector, so we rotate it
        angular_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
        corrected_angular_vel = yaw_correction[:3, :3] @ angular_vel
        corrected_msg.twist.twist.angular.x = float(corrected_angular_vel[0])
        corrected_msg.twist.twist.angular.y = float(corrected_angular_vel[1])
        corrected_msg.twist.twist.angular.z = float(corrected_angular_vel[2])
        
        # Linear velocity also needs rotation
        linear_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        corrected_linear_vel = yaw_correction[:3, :3] @ linear_vel
        corrected_msg.twist.twist.linear.x = float(corrected_linear_vel[0])
        corrected_msg.twist.twist.linear.y = float(corrected_linear_vel[1])
        corrected_msg.twist.twist.linear.z = float(corrected_linear_vel[2])
        
        # Copy twist covariance
        corrected_msg.twist.covariance = msg.twist.covariance
        
        return corrected_msg


def main(args=None):
    rclpy.init(args=args)
    node = OdomEnuCorrectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

