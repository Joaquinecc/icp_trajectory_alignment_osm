"""
Node for converting odometry to GPS coordinates.
It subscribes to odometry and initial GPS data and publishes the GPS coordinates.
"""

#ROS2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

#Lanelet2
import lanelet2
from lanelet2.core import BasicPoint3d


class Odom2GpsNode(Node):

    def __init__(self):
        super().__init__("odom2gps_node")
        self.get_logger().info("Odom2Gps node initialized")
        
        # Declare parameters
        self.declare_parameter('gps_topic', '/kitti/oxts/gps')
        self.declare_parameter('odom_topic', '/osm_align/odom')
        
        # Get parameters
        self.gps_topic: str = self.get_parameter('gps_topic').get_parameter_value().string_value
        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        
        self.get_logger().info(f"Parameters:\n"
                                f"  gps_topic: {self.gps_topic}\n"
                                f"  odom_topic: {self.odom_topic}")

        # Initialize variables
        self._utm_projector = None  # UTM projector
        self.gps_frame_id = None  # GPS frame ID from initial GPS message

        # Subscribe to initial GPS for UTM projector initialization
        self.sub_gps = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.initial_gps_callback,
            10
        )
        
        # Subscribe to odometry
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # Publish GPS coordinates
        self.pub_gps = self.create_publisher(NavSatFix, 'osm_align/gps', 10)

    def initial_gps_callback(self, msg: NavSatFix) -> None:
        """
        Initialize UTM projector from initial GPS message.
        
        Parameters
        ----------
        msg : sensor_msgs.msg.NavSatFix
            Initial GPS message.
        """
        lat0 = float(msg.latitude)
        lon0 = float(msg.longitude)
        alt0 = float(msg.altitude)
        
        self.gps_frame_id = msg.header.frame_id
        
        # Initialize UTM projector
        self._utm_projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(lat0, lon0)
        )
        self.get_logger().info(
            f"Initialized UTM projector with origin lat={lat0:.8f}, lon={lon0:.8f}, alt={alt0:.2f}"
        )
        
        # Destroy the subscription to the GPS topic, only needed for initialization
        self.destroy_subscription(self.sub_gps)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Convert odometry pose to GPS coordinates and publish.
        
        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Odometry message.
        """
        if self._utm_projector is None:
            self.get_logger().warn("UTM projector not initialized yet, waiting for initial GPS...")
            return

        if self.gps_frame_id is None:
            self.get_logger().warn("GPS frame ID not set yet, waiting for initial GPS...")
            return

        # Convert pose to GPS coordinates
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        gp = self._utm_projector.reverse(BasicPoint3d(x, y, 0.0))
        
        # Create and publish GPS message
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = self.gps_frame_id
        gps_msg.latitude = gp.lat
        gps_msg.longitude = gp.lon
        gps_msg.altitude = gp.alt
        self.pub_gps.publish(gps_msg)

    def publish_corrected_gps(self, pose) -> None:
        """
        Receive a pose object and using the UTM projector, convert it to gps coordinates.
        Publish the gps coordinates.
        
        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            Pose to publish.

        Notes
        -----
        The pose is published as a NavSatFix message.
        """
        if self._utm_projector is None:
            self.get_logger().warn("UTM projector not initialized yet, waiting for initial GPS...")
            return

        x = pose.position.x
        y = pose.position.y
        gp = self._utm_projector.reverse(BasicPoint3d(x, y, 0.0))
        
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.gps_frame_id
        msg.latitude = gp.lat
        msg.longitude = gp.lon
        msg.altitude = gp.alt
        self.pub_gps.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Odom2GpsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
