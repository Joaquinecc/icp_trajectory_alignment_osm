#ROS2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from inertiallabs_msgs.msg import InsData   
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
#Lanelet2
import lanelet2
from lanelet2.core import GPSPoint

from scipy.spatial.transform import Rotation
import numpy as np
from typing import List, Optional



INS_TOPIC = "/Inertial_Labs/ins_data"
GT_INS_TOPIC = "osm_align/gt_ins_data"
MAP_MARKER_TOPIC = "osm_align/map_markers"
class HelsinkiNode(Node):
    def __init__(self):
        super().__init__("helsinki_node")
        self.get_logger().info("Helsinki node initialized")
        self.declare_parameter('map_lanelet_path', '')


        self.map_lanelet_path: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.get_logger().info(f"Map lanelet path: {self.get_parameter('map_lanelet_path').get_parameter_value().string_value}")
        if self.map_lanelet_path == '':
            self.get_logger().error("Map lanelet path is not set")
            return
        # Initialize UTM projector lazily on first message using first LLH as origin
        self._utm_projector = None
        self._utm_origin = None  # (lat, lon, alt)

        #Map settings
        self.line_width= 0.2
        self.color_r = 188.0 / 255.0
        self.color_g = 203.0 / 255.0
        self.color_b = 169.0 / 255.0
        self.color_a = 1.0
        # Subscribers
        self.ins_sub = self.create_subscription(
            InsData,
            INS_TOPIC,
            self.ins_callback,
            10
        )
        self.gt_ins_pub = self.create_publisher(
            Odometry,
            GT_INS_TOPIC,
            10
        )
        

        # Publisher with transient local QoS (latching-like)
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.lanelet_pub = self.create_publisher(MarkerArray, MAP_MARKER_TOPIC, qos)

    def ins_callback(self, msg: InsData):
        # Initialize projector with the first received LLH as origin
        if self._utm_projector is None:
            lat0 = float(msg.llh.x)
            lon0 = float(msg.llh.y)
            alt0 = float(msg.llh.z)
            self._utm_origin = (lat0, lon0, alt0)
            self._utm_projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(lat0, lon0, alt0)
            )
            self.get_logger().info(
                f"Initialized UTM projector with origin lat={lat0:.8f}, lon={lon0:.8f}, alt={alt0:.2f}"
            )
            self._load_lanelet_map()
            self.publish_lanelet_markers()

        # Project current LLH to UTM meters
        gps_point = GPSPoint(float(msg.llh.x), float(msg.llh.y), float(msg.llh.z))
        utm_point = self._utm_projector.forward(gps_point)

        # Convert YPR (deg) to quaternion (x, y, z, w) assuming yaw->Z, pitch->Y, roll->X
        yaw_deg = float(msg.ypr.x)
        pitch_deg = float(msg.ypr.y)    
        roll_deg = float(msg.ypr.z)
        quat_xyzw = Rotation.from_euler('xyz', [yaw_deg, pitch_deg, roll_deg], degrees=True).as_quat()

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "odom"
        odom.pose.pose.position.x = float(utm_point.x)
        odom.pose.pose.position.y = float(utm_point.y)
        odom.pose.pose.position.z = float(utm_point.z)
        odom.pose.pose.orientation.x = float(quat_xyzw[0])
        odom.pose.pose.orientation.y = float(quat_xyzw[1])
        odom.pose.pose.orientation.z = float(quat_xyzw[2])
        odom.pose.pose.orientation.w = float(quat_xyzw[3])
        self.gt_ins_pub.publish(odom)

    def _load_lanelet_map(self) -> None:
        self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self._utm_projector)
        self._extract_centerlines()
        self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")

    def _extract_centerlines(self) -> List[np.ndarray]:
        centerlines: List[np.ndarray] = []  
        for lanelet in self.lanelet_map.laneletLayer:
            points_xy: List[np.ndarray] = []
            for pt in lanelet.centerline:
                xy = np.array([pt.x, pt.y])
                points_xy.append(xy)
            centerlines.append(np.array(points_xy))
        self.centerlines = centerlines
    
    def publish_lanelet_markers(self):
        markers = MarkerArray()
        for idx, line in enumerate(self.centerlines):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            # marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'lanelet_centerlines'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.line_width)
            marker.color.r = float(self.color_r)
            marker.color.g = float(self.color_g)
            marker.color.b = float(self.color_b)
            marker.color.a = float(self.color_a)
            # Convert to geometry_msgs/Point list, z=0
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in line]
            # Infinite lifetime; with transient local, late subscribers will receive
            marker.lifetime = Duration(seconds=0).to_msg()
            markers.markers.append(marker)
        self.get_logger().info(f"Published {len(markers.markers)} lanelet centerlines")
        self.lanelet_pub.publish(markers)
def main(args=None):
    rclpy.init(args=args)
    node = HelsinkiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()