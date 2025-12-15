"""
Node for odometry correction against lanelet centerlines.
It subscribes to odometry and gps data and publishes the corrected odometry and gps data.
It also publishes lanelet markers for RVIZ visualization.
It also publishes the corrected gps position in geojson format for web visualization.
"""

#ROS2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
#Lanelet2
import lanelet2
from lanelet2.core import GPSPoint, BasicPoint3d

#Python libraries
from scipy.spatial.transform import Rotation
import numpy as np
from typing import List, Optional
import json
import os

#Local libraries    
from osm_align.utils import utils
from osm_align.core.odometry_correction import OdomCorrector


MAP_MARKER_TOPIC = "osm_align/map_markers" #Topic for lanelet markers
ODOM_ALIGNED_TOPIC = 'osm_align/odom' #Topic for aligned odometry
GPS_CORRECTED_TOPIC = 'osm_align/gps' #Topic for odom corrected in gps string

class LaneCorrectionNode(Node):

    def __init__(self):
        super().__init__("lane_correction_node")
        self.get_logger().info("Lane correction node initialized")
        self.declare_parameter('matrix_lane_points', '')
        self.declare_parameter('odom_topic', '/liodom/odom')
        self.declare_parameter('initial_gps_topic', '/Inertial_Labs/initial_gps')
        self.declare_parameter('save_resuts_path', '/tmp/osm_align_results/')
        self.declare_parameter('parameters_correction', '{"pose_segment_size": 100, "knn_neighbors": 10, "valid_correspondence_threshold": 0.9, "icp_error_threshold": 2.0, "trimming_ratio": 0.2, "min_distance_threshold": 10.0}')


        self.parameters_correction: dict = json.loads(self.get_parameter('parameters_correction').get_parameter_value().string_value)
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.matrix_lane_points: str = self.get_parameter('matrix_lane_points').get_parameter_value().string_value
        self.save_resuts_path: str = self.get_parameter('save_resuts_path').get_parameter_value().string_value
        self.initial_gps_topic: str = self.get_parameter('initial_gps_topic').get_parameter_value().string_value
        self.get_logger().info(f"Parameters:\n"
                                f"  odom_topic: {self.odom_topic}\n"
                                f"  matrix_lane_points: {self.matrix_lane_points}\n"
                                f"  save_resuts_path: {self.save_resuts_path}\n"
                                f"  parameters_correction: {self.parameters_correction}\n"
                                f"  initial_gps_topic: {self.initial_gps_topic}")



        # Initialize variables
        self._utm_projector = None # UTM projector
        self._utm_origin = None  # UTM origin (lat, lon, alt)
        self.frame_count = 0 # Frame count
        self.poses_history = [] # Pose history
        self.trajectory_correction: Optional[OdomCorrector] = None #Trajectory correction
        #QOS Settings
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )


        #Initialize UTM projector and lanelet map
        self.sub_gps = self.create_subscription(
            NavSatFix,
            self.initial_gps_topic,
            self.initial_gps_callback,
            10
        )
        #Main processing loop
        self.sub_odom = self.create_subscription(   
                Odometry,
                self.odom_topic,
                self.odom_callback,
                10
            )

        #Publish results
        self.pub_lanelet_markers = self.create_publisher(MarkerArray, MAP_MARKER_TOPIC, qos)
        #Publish corrected odometry
        self.pub_corrected_odom=self.create_publisher(Odometry, ODOM_ALIGNED_TOPIC, 10) 
        #Publish corrected gps position
        self.pub_corrected_gps = self.create_publisher(NavSatFix, GPS_CORRECTED_TOPIC, 10)
    
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_to_map=utils.tf_matrix_from(self.tf_buffer, "odom", "map",timeout_sec=1.0)
        self.get_logger().info(f"tf_to_map: {self.tf_to_map}")

    def odom_callback(self, msg: Odometry) -> None:  
        if self._utm_projector is None:
            self.get_logger().warn("UTM projector not initialized yet, waiting for initial GPS...")

        pose_received=self.tf_to_map@utils.pose_to_4x4(msg.pose.pose)
        if self.trajectory_correction: 
            pose_corrected, message_code=self.trajectory_correction.apply(pose_received)

            message_str=self.trajectory_correction.get_message_str(message_code)
        else:
            pose_corrected=pose_received
            message_str="Not initialized"
        self.get_logger().info(f"frame {self.frame_count} {message_str}")
        self.frame_count += 1

        # Record pose to history before publishing
        pose_received=msg.pose.pose
        pose_received.position.x=pose_corrected[0, -1]
        pose_received.position.y=pose_corrected[1, -1]
        quat_xyzw=Rotation.from_matrix(pose_corrected[:3,:3]).as_quat()

        pose_received.orientation.x=quat_xyzw[0]
        pose_received.orientation.y=quat_xyzw[1]
        pose_received.orientation.z=quat_xyzw[2]
        pose_received.orientation.w=quat_xyzw[3]
        self.publish_odom(pose_received, msg.child_frame_id)

        if self._utm_projector:
            self.publish_corrected_gps(pose_received)
            self.poses_history.append(utils.pose_to_4x4(pose_received))

   
    def initial_gps_callback(self, msg: NavSatFix) -> None:

        lat0 = float(msg.latitude)
        lon0 = float(msg.longitude)
        alt0 = float(msg.altitude)
        
        self.gps_frame_id = msg.header.frame_id
        #Initialize UTM
        self._utm_origin = (lat0, lon0)
        self._utm_projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(lat0, lon0)
        )
        self.get_logger().info(
            f"Initialized UTM projector with origin lat={lat0:.8f}, lon={lon0:.8f}, alt={alt0:.2f}"
        )


        if self.matrix_lane_points:
            loaded= np.load(self.matrix_lane_points)
            self.points_lane_map = loaded['points_lane_map']
            gps_origin_map = loaded['origin_gps']
            self.get_logger().info(f"gps_origin_map: {gps_origin_map}")
            map_projector = lanelet2.projection.UtmProjector(
                    lanelet2.io.Origin(gps_origin_map[0], gps_origin_map[1])
                )
            offset_xy=map_projector.forward(GPSPoint(lat0, lon0))
            offset_xy=np.array([offset_xy.x, offset_xy.y])
            self.get_logger().info(f"offset_xy: {offset_xy}")
            #Update lane points, to new origin.
            self.points_lane_map[:,:2]=self.points_lane_map[:,:2]-offset_xy
            self.points_lane_map[:,2:4]=self.points_lane_map[:,2:4]-offset_xy

        self._initialize_odom_correction()

        self.destroy_subscription(self.sub_gps) #Destroy the subscription to the gps topic, only needed for initialization


    def publish_corrected_gps(self, pose) -> None:
        """
        Receive a pose object and using the UTM projector, convert it to gps coordinates.
        Publish the gps coordinates in geojson format for web visualization.
        
        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            Pose to publish.

        Notes
        -----
        The pose is published as a GeoJSON point.
        """

        x = pose.position.x
        y = pose.position.y
        gp = self._utm_projector.reverse(BasicPoint3d(x, y, 0.0))   
        
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.gps_frame_id
        msg.latitude = gp.lat
        msg.longitude = gp.lon
        msg.altitude = gp.alt
        self.pub_corrected_gps.publish(msg)

    def publish_odom(self, pose, base_frame_id: str) -> None:

        odom_msg = Odometry()
        odom_msg.header.frame_id = "map" 
        odom_msg.child_frame_id = base_frame_id
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose=pose
        self.pub_corrected_odom.publish(odom_msg)
        self.get_logger().debug(f"frame {self.frame_count} pose: {pose.position.x}, {pose.position.y}, {pose.position.z}")
        
  
    def _initialize_odom_correction(self) -> None:
        """
        Initialize the OdomCorrector object.
        """
        args={
            'pose_segment_size': self.parameters_correction['pose_segment_size'],
            'knn_neighbors': self.parameters_correction['knn_neighbors'],
            'valid_correspondence_threshold': self.parameters_correction['valid_correspondence_threshold'],
            'icp_error_threshold': self.parameters_correction['icp_error_threshold'],
            'trimming_ratio': self.parameters_correction['trimming_ratio'],
            'min_distance_threshold': self.parameters_correction['min_distance_threshold'],
        }
        self.trajectory_correction=OdomCorrector(self.points_lane_map, args)
        self.get_logger().info(f"OdomCorrector initialized")

    def save_results(self) -> None:
        """Save pose history and alignment runtimes if path is provided."""
        if not self.save_resuts_path or not self.save_resuts_path.strip():
            return
        try:
            self.save_resuts_path=os.path.join(self.save_resuts_path)
            os.makedirs(self.save_resuts_path, exist_ok=True)
            poses_path = os.path.join(self.save_resuts_path, 'poses.txt')
            with open(poses_path, 'w') as f:
                for M in self.poses_history:
                    vals = M.reshape(-1)
                    f.write(' '.join(f'{v:.12f}' for v in vals) + '\n')
            self.get_logger().info(f"Saved results to: {self.save_resuts_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to save results: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneCorrectionNode()

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
