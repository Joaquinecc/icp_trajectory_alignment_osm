#ROS2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
#Lanelet2
import lanelet2
from lanelet2.core import GPSPoint, BasicPoint3d
from sensor_msgs.msg import NavSatFix, NavSatStatus
import json
from std_msgs.msg import String
import numpy as np
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation

class MapOsmViz(Node):
    def __init__(self):
        super().__init__("map_osm_viz")
        self.get_logger().info("Map OSM Viz initialized")
        self.declare_parameter('map_lanelet_path', '')
        self.declare_parameter('gps_topic', '')
        self.declare_parameter('odom_topic1', '')
        self.declare_parameter('odom_topic2', '')
        self.declare_parameter('odom_topic3', '')
 

        self.map_lanelet_path: str = self.get_parameter('map_lanelet_path').get_parameter_value().string_value
        self.gps_topic: str = self.get_parameter('gps_topic').get_parameter_value().string_value
        self.odom_topic1: str = self.get_parameter('odom_topic1').get_parameter_value().string_value
        self.odom_topic2: str = self.get_parameter('odom_topic2').get_parameter_value().string_value
        self.odom_topic3: str = self.get_parameter('odom_topic3').get_parameter_value().string_value

        self.get_logger().info(f"Map lanelet path: {self.map_lanelet_path}")
        self.get_logger().info(f"GPS topic: {self.gps_topic}")
        self.get_logger().info(f"Odom topic 1: {self.odom_topic1}")
        self.get_logger().info(f"Odom topic 2: {self.odom_topic2}")
        self.get_logger().info(f"Odom topic 3: {self.odom_topic3}")

        self.flag_first_gps = False
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            10
        )

        # rotation_angle=np.radians(59.0)
        # self.tf_to_utm= np.identity(3)
        # self.tf_to_utm[:2,:2] = np.array([
        #     [np.cos(rotation_angle), -np.sin(rotation_angle)],
        #     [np.sin(rotation_angle),  np.cos(rotation_angle)],
        # ])


                
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)


        self.tf_odom_to_utm = self.get_transform_matrix_from_tf(
            source_frame='odom', 
            target_frame='utm', 
            timeout_sec=3
        )
        self.odom_sub1 = self.create_subscription(Odometry, self.odom_topic1, lambda msg: self.odom_callback(msg, self.odom_topic1), 10)
        self.odom_sub2 = self.create_subscription(Odometry, self.odom_topic2, lambda msg: self.odom_callback(msg, self.odom_topic2), 10)
        self.odom_sub3 = self.create_subscription(Odometry, self.odom_topic3, lambda msg: self.odom_callback(msg, self.odom_topic3), 10)
        self.gps_pub1=self.create_publisher(String, "/osm_align/map/odom_gps1", 10)
        self.gps_pub2=self.create_publisher(String, "/osm_align/map/odom_gps2", 10)
        self.gps_pub3=self.create_publisher(String, "/osm_align/map/odom_gps3", 10)


  
    def get_transform_matrix_from_tf(
        self, 
        source_frame: str = "base_link", 
        target_frame: str = "velo_link", 
        timeout_sec: float = 2.0
    ) -> np.ndarray:
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
        transform_stamped = self.tf_buffer.lookup_transform(
            target_frame,     # target frame (to)
            source_frame,     # source frame (from)
            rclpy.time.Time(seconds=0),  # latest available
            timeout=rclpy.duration.Duration(seconds=timeout_sec)
        )    
        self.get_logger().info(f"Transform stamped: {transform_stamped}")
        
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
        return transform_matrix
            
     

    def init_umt_proj_map(self,gps_msg:NavSatFix):
        self._utm_projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(gps_msg.latitude, gps_msg.longitude, gps_msg.altitude)
            )

    def gps_callback(self, gps_msg:NavSatFix):
        if not self.flag_first_gps:
            self.init_umt_proj_map(gps_msg)
            self.flag_first_gps = True
            self.get_logger().info("Initialized UTM projector")

    def odom_callback(self, odom_msg:Odometry, topic:str):
        if not self.flag_first_gps:
            return
        
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z

        point = self.tf_odom_to_utm @ np.array([x, y, z,1])  
        gp = self._utm_projector.reverse(BasicPoint3d(point[0], point[1], point[2]))   
                    # Publish small GeoJSON (Point)
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [gp.lon, gp.lat]},
            "properties": {"stamp": self.get_clock().now().to_msg().sec}
        }
        self.get_logger().info(f"Published GPS point for {topic}  ")
        if topic == self.odom_topic1:
            self.gps_pub1.publish(String(data=json.dumps(feature)))
        elif topic == self.odom_topic2:
            self.gps_pub2.publish(String(data=json.dumps(feature)))
        elif topic == self.odom_topic3:
            self.gps_pub3.publish(String(data=json.dumps(feature)))

def main(args=None):
    rclpy.init(args=args)
    node = MapOsmViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



            