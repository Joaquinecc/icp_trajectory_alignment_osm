# ROS2
from __future__ import annotations
import rclpy
from rclpy.node import Node
from inertiallabs_msgs.msg import InsData, GpsData
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus
from inertiallabs_msgs.msg import SensorData
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
from nav_msgs.msg import Odometry
from lanelet2.projection import UtmProjector
from lanelet2.core import GPSPoint
import lanelet2
from scipy.spatial.transform import Rotation
import math
# INS_TOPIC = "/gps"
GPS_TOPIC_RAW = "/Inertial_Labs/gps_data"
RENAMED_GPS_TOPIC = "/Inertial_Labs/gps_data_raw_std"
INS_TOPIC = "/Inertial_Labs/ins_data"
IMU_TOPIC = "/Inertial_Labs/sensor_data"
RENAME_TOPIC = "/Inertial_Labs/gps_data_std"
RENAME_TOPIC_IMU = "/Inertial_Labs/imu_data_std"
ODOM_INS_TOPIC = "/Inertial_Labs/odom"
class INSConversionNode(Node):
    """
    Node to convert GPS data to NavSatFix format.
    """

    def __init__(self) -> None:
        super().__init__("ins_conversion_node")

        # Declare parameters for scaling factors
        # Gyro scaling factor KG (depends on sensor range configuration)
        # 450 deg/s → KG=50, 950 deg/s → KG=20, 2000 deg/s → KG=10
        self.declare_parameter('gyro_scale_factor', 1.0)  # Default for 950 deg/s range
        
        # Accelerometer scaling factor KA (depends on sensor range configuration)
        # 8g → KA=4000, 15g → KA=2000, 40g → KA=500
        self.declare_parameter('accel_scale_factor', 1.0)  # Default for 15g range
        
        # Base frame id parameter
        self.declare_parameter('base_frame_id', 'base_link')
        
        self.KG = self.get_parameter('gyro_scale_factor').value
        self.KA = self.get_parameter('accel_scale_factor').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        
        # Conversion constants
        self.DEG_TO_RAD = math.pi / 180.0
        self.G_TO_MS2 = 9.80665  # Standard gravity in m/s²

   
        # Subscribers
        self.ins_sub = self.create_subscription(
             InsData, INS_TOPIC, self.ins_callback, 10
        )
        self.gps_sub = self.create_subscription(
            GpsData, GPS_TOPIC_RAW, self.gps_raw_callback, 10
        )
        self.imu_sub = self.create_subscription(
            SensorData, IMU_TOPIC, self.imu_callback, 10
        )
        self.imu_pub = self.create_publisher(Imu, RENAME_TOPIC_IMU, 10)
        self.odom_pub = self.create_publisher(Odometry, ODOM_INS_TOPIC, 10)
        self.gps_pub = self.create_publisher(NavSatFix, RENAME_TOPIC, 10)
        self.gps_raw_pub = self.create_publisher(NavSatFix, RENAMED_GPS_TOPIC, 10)
        # Static TF broadcaster for odom to odom_enu transform
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        # Dynamic TF broadcaster for odom to base_link transform
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info(f"Gyro scale factor (KG): {self.KG}, Accel scale factor (KA): {self.KA}")
        self.latest_coords = None

        self.proj= None
    def gps_raw_callback(self, msg):
        """
        Convert GpsData to NavSatFix message.
        """
        navsat_fix = NavSatFix()
        navsat_fix.header.stamp = msg.header.stamp
        navsat_fix.header.frame_id = msg.header.frame_id
        navsat_fix.latitude = float(msg.llh.x)
        navsat_fix.longitude = float(msg.llh.y)
        self.gps_raw_pub.publish(navsat_fix)
    def ins_callback(self, msg):

        yaw_deg = float(msg.ypr.x)
        pitch_deg = float(msg.ypr.y)
        roll_deg = float(msg.ypr.z)
        # quats=Rotation.from_euler('zxy', [yaw_deg, pitch_deg, roll_deg],degrees=True).as_quat()
        yaw_deg = (90-yaw_deg) % 360;  #Yaw correction to  Standard Clockwise orientation
        quats=Rotation.from_euler('zxy', [yaw_deg,pitch_deg, roll_deg],degrees=True).as_quat()
        if self.proj is None:
            self.proj = UtmProjector(lanelet2.io.Origin(msg.llh.x, msg.llh.y, msg.llh.z))
            self.get_logger().info(f"Projector initialized with {msg.llh.x}, {msg.llh.y}")
        xyz_p= self.proj.forward(GPSPoint(msg.llh.x, msg.llh.y, msg.llh.z))

        # Publish static TF from odom to odom_enu using yaw
        # Create rotation quaternion from yaw only (rotation around Z-axis)
        
     
        now = self.get_clock().now().to_msg()
        linear_vel_enu=[msg.vel_enu.x, msg.vel_enu.y, msg.vel_enu.z]
        # ODOM data
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = self.base_frame_id
        odom_msg.pose.pose.position.x = float(xyz_p.x)
        odom_msg.pose.pose.position.y = float(xyz_p.y)
        odom_msg.pose.pose.position.z = float(xyz_p.z)
        odom_msg.pose.pose.orientation.x = quats[0]
        odom_msg.pose.pose.orientation.y = quats[1]
        odom_msg.pose.pose.orientation.z = quats[2]
        odom_msg.pose.pose.orientation.w = quats[3]
        odom_msg.twist.twist.linear.x = linear_vel_enu[0]
        odom_msg.twist.twist.linear.y = linear_vel_enu[1]
        odom_msg.twist.twist.linear.z = linear_vel_enu[2]
        self.odom_pub.publish(odom_msg)
        
        # Publish dynamic TF from odom to base_link
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = self.base_frame_id
        t.transform.translation.x = float(xyz_p.x)
        t.transform.translation.y = float(xyz_p.y)
        t.transform.translation.z = float(xyz_p.z)
        t.transform.rotation.x = quats[0]
        t.transform.rotation.y = quats[1]
        t.transform.rotation.z = quats[2]
        t.transform.rotation.w = quats[3]
        self.tf_broadcaster.sendTransform(t)
        # GPS data
        lat = float(msg.llh.x)
        lon = float(msg.llh.y)
        alt = float(msg.llh.z)
      
        navsat_fix = NavSatFix()    
        navsat_fix.header.stamp = now
        navsat_fix.header.frame_id =  self.base_frame_id
        navsat_fix.latitude = lat
        navsat_fix.longitude = lon
        navsat_fix.altitude = alt
        navsat_fix.status.status = NavSatStatus.STATUS_FIX
        navsat_fix.status.service = NavSatStatus.SERVICE_GPS
        self.get_logger().debug(f"Publishing GPS data: {lat}, {lon}, {alt}")
        self.gps_pub.publish(navsat_fix)

    def imu_callback(self, msg):
        """
        Convert SensorData to standard Imu message.
        
        The raw gyro and accel values from the INS sensor are scaled:
        - Gyro: raw_value = angular_rate_deg_s * KG
        - Accel: raw_value = acceleration_g * KA
        
        We need to:
        1. Divide by scaling factors to get physical units
        2. Convert gyro from deg/s to rad/s
        3. Convert accel from g to m/s²

        Drop message if any converted value is nan/inf.
        """
        imu_msg = Imu()
        imu_msg.header.stamp = msg.header.stamp
        imu_msg.header.frame_id = msg.header.frame_id
        self.KG=1
        self.KA=1

        # Angular velocity: (raw / KG) * (π/180) to convert deg/s to rad/s
        gx = (float(msg.gyro.x) / self.KG) * self.DEG_TO_RAD
        gy = (float(msg.gyro.y) / self.KG) * self.DEG_TO_RAD
        gz = (float(msg.gyro.z) / self.KG) * self.DEG_TO_RAD

        # Linear acceleration: (raw / KA) * 9.80665 to convert g to m/s²
        ax = (float(msg.accel.x) / self.KA) * self.G_TO_MS2
        ay = (float(msg.accel.y) / self.KA) * self.G_TO_MS2
        az = (float(msg.accel.z) / self.KA) * self.G_TO_MS2

        # Sanity check: Drop message if any x/y/z is nan or inf
        import math
        values = [gx, gy, gz, ax, ay, az]
        if any(math.isnan(v) or math.isinf(v) for v in values):
            self.get_logger().warn(
                f"Dropping IMU message: got nan or inf in values: "
                f"gyro({gx}, {gy}, {gz}), accel({ax}, {ay}, {az})"
            )
            return

        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        self.get_logger().debug(
            f"Publishing IMU data: "
            f"gyro({imu_msg.angular_velocity.x:.3f}, {imu_msg.angular_velocity.y:.3f}, {imu_msg.angular_velocity.z:.3f}) rad/s, "
            f"accel({imu_msg.linear_acceleration.x:.3f}, {imu_msg.linear_acceleration.y:.3f}, {imu_msg.linear_acceleration.z:.3f}) m/s²"
        )

        imu_msg.angular_velocity_covariance = [0.0004, 0, 0, 0, 0.0004, 0, 0, 0, 0.0004]
        imu_msg.linear_acceleration_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu_msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = INSConversionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
