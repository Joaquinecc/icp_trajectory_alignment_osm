# Copyright 2026 Distance Technologies Oy. For internal use only.
#
"""
Node for odometry correction against lanelet centerlines.

This node subscribes to odometry and GPS data, corrects the odometry trajectory
by aligning it with lanelet centerlines from an OSM map, and publishes the
corrected odometry. The correction uses iterative closest point (ICP) algorithms
to align trajectory segments with the map data.
"""

#ROS2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import tf2_ros
#Lanelet2
import lanelet2
from lanelet2.core import GPSPoint

#Python libraries
from scipy.spatial.transform import Rotation
import numpy as np
from typing import Optional
import json
import os

#Local libraries    
from osm_align.utils import utils
from osm_align.core.odometry_correction import OdomCorrector


ODOM_ALIGNED_TOPIC = 'osm_align/odom' #Topic for aligned odometry
GPS_CORRECTED_TOPIC = 'osm_align/gps' #Topic for odom corrected in gps string

class LaneCorrectionNode(Node):
    """
    ROS2 node for correcting odometry using lanelet centerlines.
    
    This node corrects odometry trajectories by aligning them with lanelet
    centerlines from an OSM map. It uses ICP-based algorithms to find the
    optimal alignment between trajectory segments and map data.
    
    Attributes
    ----------
    _utm_projector : Optional[lanelet2.projection.UtmProjector]
        UTM coordinate projector initialized from first GPS message.
    _utm_origin : Optional[tuple]
        GPS origin coordinates (latitude, longitude) used for UTM projection.
    frame_count : int
        Counter for processed frames.
    poses_history : list
        List of 4x4 pose matrices from corrected odometry.
    trajectory_correction : Optional[OdomCorrector]
        OdomCorrector instance for applying trajectory corrections.
    tf_to_map : np.ndarray
        4x4 transformation matrix from odom frame to map/enu frame.
    """
    
    def __init__(self):
        """
        Initialize the LaneCorrectionNode.
        
        Declares ROS2 parameters, initializes subscribers and publishers,
        and sets up the TF listener for coordinate frame transformations.
        """
        super().__init__("lane_correction_node")
        self.get_logger().info("Lane correction node initialized")
        self.declare_parameter('map_points_filepath', '')
        self.declare_parameter('odom_topic_to_correct', '/liodom/odom')
        self.declare_parameter('initial_gps_topic', '/kitti/oxts/gps')
        self.declare_parameter('save_resuts_path', '/tmp/osm_align_results/')
        self.declare_parameter('parameters_correction', '{"min_segment_size": 150, "knn_neighbors": 20, "icp_error_threshold": 1.5, "max_error_consecutive": 50}')


        self.parameters_correction: dict = json.loads(self.get_parameter('parameters_correction').get_parameter_value().string_value)
        self.odom_topic_to_correct: str = self.get_parameter('odom_topic_to_correct').get_parameter_value().string_value
        self.map_points_filepath: str = self.get_parameter('map_points_filepath').get_parameter_value().string_value
        self.save_resuts_path: str = self.get_parameter('save_resuts_path').get_parameter_value().string_value
        self.initial_gps_topic: str = self.get_parameter('initial_gps_topic').get_parameter_value().string_value
        self.get_logger().info(f"Parameters:\n"
                                f"  odom_topic_to_correct: {self.odom_topic_to_correct}\n"
                                f"  map_points_filepath: {self.map_points_filepath}\n"
                                f"  save_resuts_path: {self.save_resuts_path}\n"
                                f"  parameters_correction: {self.parameters_correction}\n"
                                f"  initial_gps_topic: {self.initial_gps_topic}")



        # Initialize variables
        self._utm_projector = None # UTM projector
        self._utm_origin = None  #  Initial GPS coordinates (lat, lon, alt)
        self.frame_count = 0 # 
        self.poses_history = [] # Pose history
        self.trajectory_correction: Optional[OdomCorrector] = None #Trajectory correction

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
                self.odom_topic_to_correct,
                self.odom_callback,
                10
            )


        #Publish corrected odometry
        self.pub_corrected_odom=self.create_publisher(Odometry, ODOM_ALIGNED_TOPIC, 10) 
        #Receive TF from odom to map
        self.tf_to_map=np.eye(4)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        try:
            self.tf_to_map = utils.tf_matrix_from(self.tf_buffer,target="enu", source="odom", timeout_sec=5.0)
            self.get_logger().info(f"tf_to_map: {self.tf_to_map}")
        except Exception as e:
            self.get_logger().error(f"Failed to get TF from 'odom' to 'enu': {e}")

        # Extract yaw angle from the tf_to_map rotation matrix (Z axis Euler)
        rot = Rotation.from_matrix(self.tf_to_map[:3,:3])
        _, _, yaw = rot.as_euler('xyz', degrees=True)
        self.get_logger().info(f"Yaw angle (Z axis, degrees) from odom to enu: {yaw:.2f}")


    def odom_callback(self, msg: Odometry) -> None:
        """
        Handle odometry messages and apply trajectory correction.
        
        Transforms the incoming odometry pose to the map frame, applies
        trajectory correction if available, and publishes the corrected
        odometry. Also records the corrected pose to history.
        
        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            Incoming odometry message to correct.
        """
        if self._utm_projector is None:
            self.get_logger().warn("UTM projector not initialized yet, waiting for initial GPS...")

        pose_received=self.tf_to_map@utils.pose_to_4x4(msg.pose.pose)
        if self.trajectory_correction: 
            pose_corrected, message_id=self.trajectory_correction.apply(pose_received)

            message_str=self.trajectory_correction._get_messages_str(message_id)
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
            self.poses_history.append(utils.pose_to_4x4(pose_received))

   
    def initial_gps_callback(self, msg: NavSatFix) -> None:
        """
        Initialize UTM projector and load lanelet map from first GPS message.
        
        Uses the first GPS message to initialize the UTM projector and load
        the lanelet map points. The map points are transformed to the new
        GPS origin, and the OdomCorrector is initialized with the map data.
        
        Parameters
        ----------
        msg : sensor_msgs.msg.NavSatFix
            Initial GPS message containing latitude, longitude, and altitude.
            
        Raises
        ------
        ValueError
            If no map points filepath is provided in parameters.
        """
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


        if self.map_points_filepath:
            loaded= np.load(self.map_points_filepath)
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
        else:
            raise ValueError("No map points filepath provided, trajectory correction not possible")
        self.destroy_subscription(self.sub_gps) #Destroy the subscription to the gps topic, only needed for initialization


    def publish_odom(self, pose, base_frame_id: str) -> None:
        """
        Publish corrected odometry message.
        
        Creates and publishes an Odometry message with the corrected pose.
        The message is published in the "map" frame.
        
        Parameters
        ----------
        pose : geometry_msgs.msg.Pose
            Corrected pose to publish.
        base_frame_id : str
            Child frame ID for the odometry message (typically the base frame).
        """
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
        
        Creates an OdomCorrector instance with the loaded lanelet map points
        and correction parameters. Only valid parameter keys are passed to
        the OdomCorrector constructor.
        """
        # Collect only known supported keys from parameters_correction dynamically
        valid_keys = [
            'min_segment_size',
            'knn_neighbors',
            'valid_correspondence_threshold',
            'icp_error_threshold',
            'trimming_ratio',
            'min_distance_threshold',
            'max_error_consecutive',
            'max_segment_size',
        ]
        args = {k: self.parameters_correction[k] for k in valid_keys if k in self.parameters_correction}
        self.trajectory_correction=OdomCorrector(self.points_lane_map, args)
        self.get_logger().info(f"OdomCorrector initialized")

    def save_results(self) -> None:
        """
        Save pose history to file if save path is provided.
        
        Writes all collected corrected poses to a text file in the specified
        save directory. Each pose is written as a flattened 4x4 matrix with
        16 space-separated floating-point values per line.
        
        Notes
        -----
        The save path is specified via the 'save_resuts_path' parameter.
        If the path is empty or not set, this method returns without saving.
        """
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
