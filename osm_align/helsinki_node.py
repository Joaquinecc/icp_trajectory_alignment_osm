#ROS2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from inertiallabs_msgs.msg import InsData   
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String

#Lanelet2
import lanelet2
from lanelet2.core import GPSPoint, BasicPoint3d

#Python libraries
from scipy.spatial.transform import Rotation
import numpy as np
from typing import List
from scipy.spatial import cKDTree
import json

#Local libraries
from osm_align.utils import utils
from osm_align import OdomCorrector





INS_TOPIC = "/Inertial_Labs/ins_data" #Topic for INS data
INS2ODOM_TOPIC = "osm_align/ins_data2_odom" #Convert to UTM
MAP_MARKER_TOPIC = "osm_align/map_markers" #Topic for lanelet markers
ODOM_ALIGNED_TOPIC = 'osm_align/odom_aligned' #Topic for aligned odometry
CAR_GEOJSON_TOPIC = 'osm_align/car_geojson' #Topic for odom corrected in gps string


class HelsinkiNode(Node):
    """
    ROS2 Node made for Helsinki demo dataset. It subscribes to odometry topic and publishes aligned odometry.
    It also publishes lanelet markers for RVIZ visualization.
    It also publishes the corrected gps position in geojson format for web visualization.

    Parameters
    ----------
    odom_topic : str
        Topic name of the odometry to correct.
    map_lanelet_path : str
        Path to the lanelet map.
    pose_segment_size : int
        Number of points in each pose segment.
    knn_neighbors : int
        Number of nearest neighbors to use for ICP.
    valid_correspondence_threshold : float
        Threshold for valid correspondence.
    icp_error_threshold : float
        Threshold for ICP error.
    trimming_ratio : float
        Ratio of points to trim from the beginning and end of the trajectory.
    min_distance_threshold : float
        Minimum distance between points in the trajectory.
    """
    def __init__(self):
        super().__init__("helsinki_node")
        self.get_logger().info("Helsinki node initialized")
        self.declare_parameter('odom_topic', '/liodom/odom')
        self.declare_parameter('map_lanelet_path', '')

        #Odom Correction Parameters
        self.declare_parameter('pose_segment_size', 100)
        self.declare_parameter('knn_neighbors', 10)
        self.declare_parameter('valid_correspondence_threshold', 0.9)
        self.declare_parameter('icp_error_threshold', 2.0)
        self.declare_parameter('trimming_ratio', 0.2)
        self.declare_parameter('min_distance_threshold', 10.0)


        self.map_lanelet_path: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        if self.map_lanelet_path == '':
            self.get_logger().error("Map lanelet path is not set")
            return

        # Odom Correction Parameters
        self.pose_segment_size: int = self.get_parameter('pose_segment_size').get_parameter_value().integer_value
        self.knn_neighbors: int = self.get_parameter('knn_neighbors').get_parameter_value().integer_value
        self.valid_correspondence_threshold: float = self.get_parameter('valid_correspondence_threshold').get_parameter_value().double_value
        self.icp_error_threshold: float = self.get_parameter('icp_error_threshold').get_parameter_value().double_value
        self.trimming_ratio: float = self.get_parameter('trimming_ratio').get_parameter_value().double_value
        self.min_distance_threshold: float = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value

        # Log info: print odom_topic and map_lanelet_path
        self.get_logger().info(f"odom_topic: {self.odom_topic}")
        self.get_logger().info(f"map_lanelet_path: {self.map_lanelet_path}")
        self.get_logger().info(f"pose_segment_size: {self.pose_segment_size}")
        self.get_logger().info(f"knn_neighbors: {self.knn_neighbors}")
        self.get_logger().info(f"valid_correspondence_threshold: {self.valid_correspondence_threshold}")
        self.get_logger().info(f"icp_error_threshold: {self.icp_error_threshold}")
        self.get_logger().info(f"trimming_ratio: {self.trimming_ratio}")
        self.get_logger().info(f"min_distance_threshold: {self.min_distance_threshold}")
        self.get_logger().info(f"Map lanelet path: {self.map_lanelet_path}")



        # Initialize variables
        self._utm_projector = None # UTM projector
        self._utm_origin = None  # UTM origin (lat, lon, alt)
        self.frame_count = 0 # Frame count


        #QOS Settings
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )


        # Subscribers
        #Subscribe to INS data 
        self.ins_sub = self.create_subscription(
            InsData,
            INS_TOPIC,
            self.ins_callback,
            10
        )
        #Subscribe to odometry data for correction
        self.odom_sub = self.create_subscription(   
                Odometry,
                self.odom_topic,
                self.odom_callback,
                10
            )

        #Publishers
        #Publish lanelet markers for RVIZ visualization
        self.lanelet_pub = self.create_publisher(MarkerArray, MAP_MARKER_TOPIC, qos)
        #Publish INS data for UTM projection
        self.ins2odom_pub = self.create_publisher(
            Odometry,
            INS2ODOM_TOPIC,
            10
        )
        #Publish aligned odometry
        self.publisher_odom=self.create_publisher(Odometry, ODOM_ALIGNED_TOPIC, 10) 
        #Publish the corrected odoometry in geojson format for web visualization
        self.geojson_pub = self.create_publisher(String, CAR_GEOJSON_TOPIC, 10)

    def odom_callback(self, msg: Odometry) -> None:  
        """
        Correct the odometry using the lanelet map.
        Publsih the corrected odoometry and the corrected gps position in geojson format for web visualization.

        Raises
        ------
        Exception
        If the odometry cannot be corrected.

        Notes
        -----
        The odometry is corrected using the lanelet map.
        """
        pose_corrected, message=self.trajectory_correction.apply(self.tf_to_utm@utils.pose_to_4x4(msg.pose.pose))
        if message==0:
            self.get_logger().debug(f"frame {self.frame_count} trajectory length < {self.min_distance_threshold}, skip ICP")
        elif message==1:
            self.get_logger().debug(f"frame {self.frame_count} ICP error > {self.icp_error_threshold}, skip ICP")
        elif message==-1:
            pass
        else:
            self.get_logger().debug(f"frame {self.frame_count} ICP error < {message}, ICP success")

        self.frame_count += 1

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
        self.publish_corrected_gps(pose_recived)
   
    def ins_callback(self, msg: InsData):
        """
        Convert the INS data to UTM and publish the aligned odometry.
        Initialize the UTM projector with the first received LLH as origin.
        Initialize the rotation matrix to convert from odom to utm coordinates.
        Load the lanelet map.
        Publish the lanelet markers for RVIZ visualization.

        Parameters
        ----------
        msg : InsData
            INS data to convert to UTM.

        """
        # Initialize projector with the first received LLH as origin
        if self._utm_projector is None:
            # self.get_logger().info(f"First INS message received {msg}")
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
            self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self._utm_projector)
            self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")



            # Initialize the transformation matrix to convert from odom to utm
            theta = np.deg2rad(150)
            self.tf_lidar_to_base_link=np.eye(4)
            # self.tf_lidar_to_base_link[:3, :3] =np.array([
            # [1, 0, 0],
            # [0, np.cos(theta), -np.sin(theta)],
            # [0, np.sin(theta), np.cos(theta)]
            # ])

            self.tf_lidar_to_base_link[:3, :3] =np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
            ])

            self.tf_to_utm=np.eye(4)

            # yaw_deg = float(msg.ypr.x)
            # pitch_deg = float(msg.ypr.y)    
            # roll_deg = float(msg.ypr.z)
            # r = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)
            # self.tf_to_utm[:3, :3] = r.as_matrix()

            self.tf_to_utm=self.tf_to_utm@self.tf_lidar_to_base_link
            self.get_logger().info(f"tf_to_utm: {self.tf_to_utm}")

            # Initialize the OdomCorrector object.
            self._initialize_odom_correction()
            # Publish the lanelet markers for RVIZ visualization
            self.publish_lanelet_markers()

        # Project current LLH to UTM meters
        gps_point = GPSPoint(float(msg.llh.x), float(msg.llh.y), float(msg.llh.z))
        utm_point = self._utm_projector.forward(gps_point)

        # Convert YPR (deg) to quaternion (x, y, z, w) assuming yaw->Z, pitch->Y, roll->X
        yaw_deg = float(msg.ypr.x)
        pitch_deg = float(msg.ypr.y)    
        roll_deg = float(msg.ypr.z)
        quat_xyzw = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True).as_quat()

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
        self.ins2odom_pub.publish(odom)   

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
        # Publish small GeoJSON (Point)
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [gp.lon, gp.lat]},
            "properties": {"stamp": self.get_clock().now().to_msg().sec}
        }

        msg_str=String()
        msg_str.data=json.dumps(feature)
        self.geojson_pub.publish(msg_str)

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
        odom_msg.child_frame_id = "base_link"
        

        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose=pose
        self.publisher_odom.publish(odom_msg)

        self.get_logger().debug(f"frame {self.frame_count} pose: {pose.position.x}, {pose.position.y}, {pose.position.z}")
        
    def publish_lanelet_markers(self):
        """
        Run only once after the lanelet map is loaded.
        Publish the lanelet markers for RVIZ visualization.
        """
        centerlines: List[np.ndarray] = []  
        for lanelet in self.lanelet_map.laneletLayer:
            points_xy: List[np.ndarray] = []
            for pt in lanelet.centerline:
                xy = np.array([pt.x, pt.y])
                points_xy.append(xy)
            centerlines.append(np.array(points_xy))
        #Map settings for RVIZ visualization
        line_width= 0.2
        color_r = 188.0 / 255.0
        color_g = 203.0 / 255.0
        color_b = 169.0 / 255.0
        color_a = 1.0
        markers = MarkerArray()
        for idx, line in enumerate(centerlines):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            marker.ns = 'lanelet_centerlines'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(line_width)
            marker.color.r = float(color_r)
            marker.color.g = float(color_g)
            marker.color.b = float(color_b)
            marker.color.a = float(color_a)
            # Convert to geometry_msgs/Point list, z=0
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in line]
            # Infinite lifetime; with transient local, late subscribers will receive
            marker.lifetime = Duration(seconds=0).to_msg()
            markers.markers.append(marker)
        self.get_logger().info(f"Published {len(markers.markers)} lanelet centerlines")
        self.lanelet_pub.publish(markers)

    def _initialize_odom_correction(self) -> None:
        """
        Initialize the OdomCorrector object.
        """
        args={
            'pose_segment_size': self.pose_segment_size,
            'knn_neighbors': self.knn_neighbors,
            'valid_correspondence_threshold': self.valid_correspondence_threshold,
            'icp_error_threshold': self.icp_error_threshold,
            'trimming_ratio': self.trimming_ratio,
            'min_distance_threshold': self.min_distance_threshold,
        }
        print(f"lanelet_map: {self.lanelet_map}")
        lane_points, lane_points_nn = utils.lane_points_and_it_nn(self.lanelet_map)
        self.trajectory_correction=OdomCorrector(lane_points, lane_points_nn, cKDTree(lane_points), args)



def main(args=None):
    rclpy.init(args=args)
    node = HelsinkiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()