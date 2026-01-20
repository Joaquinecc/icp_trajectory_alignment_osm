# Copyright 2026 Distance Technologies Oy. For internal use only.
#
"""
Node for converting odometry to GPS coordinates.

This node subscribes to odometry messages and converts the pose positions
to GPS coordinates (latitude, longitude, altitude) using a UTM projector
initialized from an initial GPS message. The converted GPS coordinates are
published as NavSatFix messages.
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
    """
    ROS2 node for converting odometry poses to GPS coordinates.
    
    This node subscribes to odometry messages and converts the 2D position
    (x, y) to GPS coordinates using a UTM projector. The UTM projector is
    initialized from the first received GPS message, which sets the origin
    for coordinate conversion.
    
    Attributes
    ----------
    _utm_projector : Optional[lanelet2.projection.UtmProjector]
        UTM coordinate projector initialized from first GPS message.
    gps_frame_id : Optional[str]
        Frame ID from the initial GPS message, used for published GPS messages.
    """
    
    def __init__(self):
        """
        Initialize the Odom2GpsNode.
        
        Declares ROS2 parameters for input topics and initializes
        subscribers and publishers.
        """
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
        
        Uses the first received GPS message to initialize the UTM projector
        with the GPS coordinates as the origin. After initialization, unsubscribes
        from the GPS topic as it's no longer needed.
        
        Parameters
        ----------
        msg : sensor_msgs.msg.NavSatFix
            Initial GPS message containing latitude, longitude, and altitude.
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
        
        Extracts the 2D position (x, y) from the odometry pose and converts
        it to GPS coordinates using the UTM projector. Publishes the result
        as a NavSatFix message.
        
        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Odometry message containing the pose to convert.
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
