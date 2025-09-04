import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
import lanelet2
import numpy as np
from typing import List, Optional

from osm_align.utils.kitti_utils import angle_dict, cordinta_dict

from tf2_ros import Buffer, TransformListener
class LaneletMapVizNode(Node):
    """
    ROS2 node that publishes Lanelet2 centerlines as visualization markers.

    - Loads the Lanelet2 map using UTM projector with origin from KITTI frame_id
    - Rotates map points by the precomputed angle correction to align with odom frame
    - Publishes a MarkerArray with LINE_STRIP markers (transient local QoS) so RViz can latch
    """

    def __init__(self) -> None:
        super().__init__('lanelet_map_viz_node')

        # Parameters
        self.declare_parameter('frame_id', '00')
        self.declare_parameter('map_lanelet_path', '')
        self.declare_parameter('marker_topic', '/osm_align/lanelet_markers')
        self.declare_parameter('line_width', 0.2)
        self.declare_parameter('min_point_spacing', 3.0)
        self.declare_parameter('color_r', 188.0 / 255.0)
        self.declare_parameter('color_g', 203.0 / 255.0)
        self.declare_parameter('color_b', 169.0 / 255.0)
        self.declare_parameter('color_a', 1.0)
        self.declare_parameter('frame', 'velo_link')

        self.frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value
        self.map_lanelet_path_param: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.marker_topic: str = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.line_width: float = self.get_parameter('line_width').get_parameter_value().double_value
        self.min_point_spacing: float = self.get_parameter('min_point_spacing').get_parameter_value().double_value
        self.color_r: float = self.get_parameter('color_r').get_parameter_value().double_value
        self.color_g: float = self.get_parameter('color_g').get_parameter_value().double_value
        self.color_b: float = self.get_parameter('color_b').get_parameter_value().double_value
        self.color_a: float = self.get_parameter('color_a').get_parameter_value().double_value
        self.frame: str = self.get_parameter('frame').get_parameter_value().string_value

        # Resolve map path and origin/angle
        self.map_lanelet_path: str = self.map_lanelet_path_param or \
            f'/home/joaquinecc/Documents/dataset/kitti/dataset/map/{self.frame_id}/lanelet2_seq_{self.frame_id}.osm'
        self.origin_coords_lanelet: List[float] = [
            cordinta_dict[self.frame_id]['origin_lat'],
            cordinta_dict[self.frame_id]['origin_lon']
        ]
        self.angle_lanelet_correction: float = angle_dict[self.frame_id]

        # Publisher with transient local QoS (latching-like)
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, qos)

        # Load and publish once
        try:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

            self.tf_to_map = self.tf_buffer.lookup_transform(
                'map',     # target frame (to)
                self.frame,     # source frame (from)
                rclpy.time.Time(seconds=0),  # latest available
                rclpy.duration.Duration(seconds=5)
                )   
            self.get_logger().info(f"Transform to map: {self.tf_to_map}")

            self._load_lanelet_map()
            centerlines = self._extract_centerlines()
            self._publish_markers(centerlines)
            self.get_logger().info(
                f"Published Lanelet2 centerlines to '{self.marker_topic}' with transient local QoS"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to publish lanelet map: {e}")

    
        

    def _load_lanelet_map(self) -> None:
        projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(self.origin_coords_lanelet[0], self.origin_coords_lanelet[1])
        )
        self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, projector)
        self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")

    def _extract_centerlines(self) -> List[np.ndarray]:
        rotation_angle = np.radians(self.angle_lanelet_correction)
        R_M = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)],
        ])

        rotation_matrix = self.tf_to_map.transform.rotation
        rotation_matrix = Rotation.from_quat([rotation_matrix.x, rotation_matrix.y, rotation_matrix.z, rotation_matrix.w])
        rotation_matrix = rotation_matrix.as_matrix()[:2, :2]
        R_M = rotation_matrix @ R_M

        centerlines: List[np.ndarray] = []
        for lanelet in self.lanelet_map.laneletLayer:
            prev_point: Optional[np.ndarray] = None
            points_xy: List[np.ndarray] = []
            for pt in lanelet.centerline:
                xy = R_M @ np.array([pt.x, pt.y])
                if prev_point is not None:
                    if np.linalg.norm(xy - prev_point) < self.min_point_spacing:
                        continue
                prev_point = xy
                points_xy.append(xy)
            if len(points_xy) >= 2:
                centerlines.append(np.array(points_xy))
        self.get_logger().info(f"Extracted {len(centerlines)} centerline polylines")
        return centerlines

    def _publish_markers(self, centerlines: List[np.ndarray]) -> None:
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

        self.marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneletMapVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
