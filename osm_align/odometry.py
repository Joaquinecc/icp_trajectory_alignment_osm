import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import math
from typing import List, Tuple
import os
import lanelet2
# Add these imports for tf2
from tf2_ros import TransformBroadcaster
import tf_transformations
# Add service imports
from geometry_msgs.msg import Transform
from liodom.srv import TransformCorrection
from scipy.spatial.transform import Rotation as R
from osm_align.utils import solve_trimmed_icp_2d,find_interception_normal_shooting_nn
from scipy.spatial import cKDTree

# Note: For Python, we'll need to use equivalent libraries
# lanelet2_core -> lanelet2
# pcl -> open3d or scipy.spatial for KD-tree
# Eigen -> numpy

# Configuration parameters
pose_history_size_ = 20
knn_neighbors_=10
valid_correspondence_threshold_=0.6
icp_error_threshold_=1
odom_topic_= '/liodom/odom'
# odom_topic_= '/kitti/gtruth/odom'
# Configurable service name for transform correction
# This service receives transform corrections and updates the laser odometry offset
transform_correction_service_ = '/liodom/transform_correction'

class OdomCorrection(Node):

    def __init__(self, map_lanelet_path: str, origin_coords_lanelet: List[float], angle_lanelet_correction: float):
        super().__init__('odom_subscriber')
        
        # Store parameters
        self.map_lanelet_path = map_lanelet_path
        self.origin_coords_lanelet = origin_coords_lanelet
        self.angle_lanelet_correction = angle_lanelet_correction
        
        # Initialize lanelet-related variables
        self.projector = None
        self.lanelet_map = None
        self.lane_kdtree = None
        self.lane_cloud = None
        self.pose_history = []
        
        
        # Load lanelet map and build KD-tree
        self._load_lanelet_map()
        self._build_lane_kdtree()
        
        # Create subscription
        self.subscription = self.create_subscription(
            Odometry,
            odom_topic_,  # Change if your odometry topic has a different name
            self.odom_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        
  
        # Create service client
        self.client = self.create_client(TransformCorrection, transform_correction_service_)       
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def _load_lanelet_map(self):

        self.projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(self.origin_coords_lanelet[0], self.origin_coords_lanelet[1])
        )
        self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self.projector)
        
        self.get_logger().info(f"Lanelet map loaded from: {self.map_lanelet_path}")
        self.get_logger().info(f"Origin coordinates: {self.origin_coords_lanelet}")
        self.get_logger().info(f"Angle correction: {self.angle_lanelet_correction} degrees")
        self.get_logger().info(f"Angle correction: asdadsas degrees")

    def _build_lane_kdtree(self):
        """Build KD-tree for lane points"""
        # Define rotation matrix
        rotation_angle = self.angle_lanelet_correction * (math.pi / 180.0)
        cos_angle = math.cos(rotation_angle)
        sin_angle = math.sin(rotation_angle)
        
        # Rotation matrix (2D)
        R_M = np.array([[cos_angle, -sin_angle],
                        [sin_angle, cos_angle]])
        
        lane_points = []
        
        for lanelet in self.lanelet_map.laneletLayer:
            centerline = lanelet.centerline
            for point in centerline:
                correct_point = R_M @ np.array([point.x, point.y])
                lane_points.append([correct_point[0], correct_point[1]])
        
        # Build KD-tree using scipy (Python equivalent of PCL KD-tree)
        lane_points = sorted(lane_points, key=lambda p: p[0]**2 + p[1]**2)
        self.lane_points = np.array(lane_points)
        self.lane_kdtree = cKDTree(self.lane_points)
        
        self.get_logger().info(f"Built KD-tree with {len(lane_points)} lane points")
    

    def _send_transform_correction(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        """
        Send a 3D transform correction to the laser odometry service.

        This method constructs and sends a TransformCorrection service request containing
        a geometry_msgs/Transform message, with translation and rotation (as quaternion).

        Parameters
        ----------
        rotation_matrix : np.ndarray
            A 3x3 NumPy array representing the rotation to apply (in world or map frame).
        translation_vector : np.ndarray
            A 3-element NumPy array representing the translation to apply (in meters).

        Returns
        -------
        None

        Notes
        -----
        - The orientation difference is sent as a quaternion [x, y, z, w].
        - The service must be available and expect the correct message types.
        """
        req = TransformCorrection.Request()
        # geometry_msgs/Transform expects translation (x, y, z) and rotation (x, y, z, w)

        transform = Transform()
        transform.translation.x = float(translation_vector[0])
        transform.translation.y = float(translation_vector[1])
        transform.translation.z = float(translation_vector[2])
        quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        transform.rotation.x = float(quat[0])
        transform.rotation.y = float(quat[1])
        transform.rotation.z = float(quat[2])
        transform.rotation.w = float(quat[3])
        # print(f"x and y: {transform.translation.x}, {transform.translation.y}")
        req.transform = transform
        self.client.call_async(req)

    def align_pose(self):
        """
        Align the current pose history to the lanelet map using ICP.

        This method:
        - Extracts the (x, y) positions from the pose history.
        - Finds nearest lanelet map points for each trajectory point.
        - Filters out invalid correspondences (e.g., NaN projections).
        - Runs 2D ICP to estimate the best-fit rigid transformation (rotation and translation)
          between the trajectory and the map.
        - If ICP converges with sufficiently low error, applies the transformation to the
          pose history (in-place), updating the x and y coordinates of each pose.

        Returns
        -------
        None

        Logs
        ----
        - Number of valid correspondences found.
        - ICP final error.
        - Whether ICP alignment succeeded or failed.

        Notes
        -----
        - Only x and y coordinates are transformed; z remains unchanged.
        - The transformation is only applied if the ICP error is below a configured threshold.
        """
        trajectory_points = [[pose.position.x, pose.position.y] for pose in self.pose_history]
        trajectory_points = np.array(trajectory_points)

        knn_index = self.lane_kdtree.query(trajectory_points, k=10)[1]
        best_lane_points = find_interception_normal_shooting_nn(trajectory_points, knn_index, self.lane_points)
        valid_mask = ~np.isnan(best_lane_points).any(axis=1)

        best_lane_points = best_lane_points[valid_mask]
        trajectory_points = trajectory_points[valid_mask]

        # self.get_logger().info(
        #     f"Found {len(trajectory_points)} valid correspondences out of {len(self.pose_history)} trajectory points"
        # )
        if len(trajectory_points) < len(self.pose_history) * valid_correspondence_threshold_:
            self.get_logger().info("Not enough valid correspondences for ICP alignment")
            return

        R_total, T_total, final_error = solve_trimmed_icp_2d(trajectory_points, best_lane_points,)

        if final_error < icp_error_threshold_:
            self.get_logger().info(f"Pass threshold, ICP final error: {final_error}")

            pose_history = []
            self.pose_history = np.array(self.pose_history)
            # Build a 3D transformation matrix that only affects x and y
            R_3d = np.eye(3)
            R_3d[:2, :2] = R_total
            T_3d = np.zeros(3)
            T_3d[:2] = T_total
            for pose in self.pose_history:
                point_xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
                point_xyz = R_3d @ point_xyz + T_3d
                pose.position.x = point_xyz[0]
                pose.position.y = point_xyz[1]
                # pose.position.z remains unchanged
                pose_history.append(pose)
            self.pose_history = pose_history
            self._send_transform_correction(R_3d, T_3d)
        else:
            self.get_logger().info(f"Fail threshold, ICP final error: {final_error}")
        


    def odom_callback(self, msg: Odometry):
        # Store both raw and corrected poses
        self.pose_history.append(msg.pose.pose)
        if len(self.pose_history) == pose_history_size_:
            self.align_pose()
            self.pose_history.pop(0)




def main(args=None):
    rclpy.init(args=args)

    # Default parameters - you can modify these or pass them as command line arguments
    map_lanelet_path = '/home/joaquinecc/Documents/dataset/kitti/dataset/map/02/lanelet2_seq_02.osm' # Update this path
    origin_coords_lanelet = [48.987607723096, 8.4697469732634]  # [lat, lon] or [x, y] depending on your coordinate system
    angle_lanelet_correction = -53.5  # degrees
    
    
    node = OdomCorrection(map_lanelet_path, origin_coords_lanelet, angle_lanelet_correction)
    
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
