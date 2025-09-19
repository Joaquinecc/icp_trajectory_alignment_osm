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
from osm_align import OdomCorrector

from scipy.spatial.transform import Rotation
import numpy as np
from typing import List, Optional
from scipy.spatial import cKDTree
from osm_align.utils import utils



INS_TOPIC = "/Inertial_Labs/ins_data"
GT_INS_TOPIC = "osm_align/gt_ins_data"
MAP_MARKER_TOPIC = "osm_align/map_markers"


class HelsinkiNode(Node):
    def __init__(self):
        super().__init__("helsinki_node")
        self.get_logger().info("Helsinki node initialized")
        self.declare_parameter('map_lanelet_path', '')
        self.declare_parameter('pose_segment_size', 100)
        self.declare_parameter('knn_neighbors', 10)
        self.declare_parameter('valid_correspondence_threshold', 0.9)
        self.declare_parameter('icp_error_threshold', 2.0)
        self.declare_parameter('trimming_ratio', 0.2)
        self.declare_parameter('min_distance_threshold', 10.0)
        self.declare_parameter('odom_topic', '/liodom/odom')


        self.map_lanelet_path: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.get_logger().info(f"Map lanelet path: {self.get_parameter('map_lanelet_path').get_parameter_value().string_value}")
        if self.map_lanelet_path == '':
            self.get_logger().error("Map lanelet path is not set")
            return
        # Initialize UTM projector lazily on first message using first LLH as origin
        self._utm_projector = None
        self._utm_origin = None  # (lat, lon, alt)

        #Odom Correction variables
        self.frame_count = 0
        self.poses_history = []

        #Odom Correction Parameters
        self.pose_segment_size: int = self.get_parameter('pose_segment_size').get_parameter_value().integer_value
        self.knn_neighbors: int = self.get_parameter('knn_neighbors').get_parameter_value().integer_value
        self.valid_correspondence_threshold: float = self.get_parameter('valid_correspondence_threshold').get_parameter_value().double_value
        self.icp_error_threshold: float = self.get_parameter('icp_error_threshold').get_parameter_value().double_value
        self.trimming_ratio: float = self.get_parameter('trimming_ratio').get_parameter_value().double_value
        self.min_distance_threshold: float = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value

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



        # Create subscription
        self.subscription = self.create_subscription(   
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        self.publisher_odom=self.create_publisher(Odometry, '/osm_align/odom_aligned', 10) 
    
    def odom_callback(self, msg: Odometry) -> None:  
        pose_corrected, message=self.trajectory_correction.apply(utils.pose_to_4x4(msg.pose.pose))
        if message==0:
            self.get_logger().debug(f"frame {self.frame_count} trajectory length < {self.min_distance_threshold}, skip ICP")
        elif message==1:
            self.get_logger().debug(f"frame {self.frame_count} ICP error > {self.icp_error_threshold}, skip ICP")
        elif message==-1:
            pass
        else:
            self.get_logger().debug(f"frame {self.frame_count} ICP error < {message}, ICP success")

        self.frame_count += 1
        self.poses_history.append(pose_corrected)

        # Record pose to history before publishing
        pose_recived=msg.pose.pose
        pose_recived.position.x=pose_corrected[0, -1]
        pose_recived.position.y=pose_corrected[1, -1]
        pose_recived.position.z=pose_corrected[2, -1]
        pose_recived.orientation.x=pose_corrected[0, 0]
        pose_recived.orientation.y=pose_corrected[1, 0]
        pose_recived.orientation.z=pose_corrected[2, 0]
        pose_recived.orientation.w=pose_corrected[3, 0]
        self.publish_odom(pose_recived)

 
    def publish_odom(self, pose) -> None:
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
        self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")


        args={
            'pose_segment_size': self.pose_segment_size,
            'knn_neighbors': self.knn_neighbors,
            'valid_correspondence_threshold': self.valid_correspondence_threshold,
            'icp_error_threshold': self.icp_error_threshold,
            'trimming_ratio': self.trimming_ratio,
            'min_distance_threshold': self.min_distance_threshold,
        }
        lane_points, lane_points_next = self._build_lane_kdtree_points()
        self.trajectory_correction=OdomCorrector(lane_points, lane_points_next, cKDTree(lane_points), args)

    def _build_lane_kdtree_points(self) -> None:

        min_dist = 2.0
        lane_points_next = []
        lane_points = []
        
        for lanelet in self.lanelet_map.laneletLayer:
            prev_point = None
            centerline = lanelet.centerline
            aux_points = []
            for i, point in enumerate(centerline):
                corrected_point = np.array([point.x, point.y])
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



        self.get_logger().info(f"Built KD-tree with {len(lane_points)} lane points")
        return lane_points, lane_points_next

    def _extract_centerlines(self) -> List[np.ndarray]:
        centerlines: List[np.ndarray] = []  
        for lanelet in self.lanelet_map.laneletLayer:
            points_xy: List[np.ndarray] = []
            for pt in lanelet.centerline:
                xy = np.array([pt.x, pt.y])
                points_xy.append(xy)
            centerlines.append(np.array(points_xy))
        return centerlines
    
    def publish_lanelet_markers(self):
        centerlines = self._extract_centerlines()
        markers = MarkerArray()
        for idx, line in enumerate(centerlines):
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