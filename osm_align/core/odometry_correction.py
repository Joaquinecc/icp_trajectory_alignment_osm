"""Odometry correction module using 2D trimmed ICP against lanelet centerlines."""
import numpy as np
from typing import List, Optional, Dict, Any
from scipy.spatial import cKDTree
# from geometry_msgs.msg import Pose  # Not used here; keep minimal deps
try:
    import osm_align.utils.utils as utils
except:
    import utils.utils as utils
import lanelet2
import time
# Configuration parameters are now declared as ROS parameters in the node

MAX_ERROR_CONSECUTIVE=30      

class OdomCorrector():
    """
    Dataset-agnostic odometry trajectory corrector using 2D trimmed ICP.

    This class accumulates a sliding window of past poses and periodically
    aligns the vehicle trajectory to lanelet centerline points using a
    trimmed ICP variant. It is designed to be reused across datasets
    (e.g., KITTI, custom logs) as long as lanelet-like centerline points
    and their spatial index are provided.

    Parameters
    ----------
    lane_points : numpy.ndarray or None
        Array of shape (N, 2) with lane centerline points in meters.
    lane_points_neighbour : numpy.ndarray or None
        Array of shape (N, 2) with "next" point associations for each
        centerline point, used to compute local tangents.
    lane_kdtree : scipy.spatial.cKDTree or None
        KD-tree built over `lane_points` for fast nearest-neighbor queries.
    args : Dict[str, Any]
        Configuration dictionary with the following keys:
        - 'pose_segment_size' (int): size of the sliding window
        - 'knn_neighbors' (int): number of neighbors for correspondence
        - 'valid_correspondence_threshold' (float): min fraction of valid
          correspondences required to run ICP
        - 'icp_error_threshold' (float): maximum ICP error to accept update
        - 'trimming_ratio' (float): fraction of largest residuals to trim
        - 'min_distance_threshold' (float): minimum path length in the
          current window to trigger alignment
    """

    def __init__(
        self,
        points_lane_map, #is a 6xN array of points
        args: Dict[str, Any],
    ) -> None:

        self.lane_points=points_lane_map[:,0:2]
        self.lane_points_neighbour=points_lane_map[:,2:4] #To get intersections
        self.lane_direction_tan =points_lane_map[:,4:6] #Car direction


        self.lane_kdtree: Optional[cKDTree] = cKDTree(self.lane_points)

        self.pose_segment_size: int = args['pose_segment_size']
        self.knn_neighbors: int = args['knn_neighbors']
        self.valid_correspondence_threshold: float = args['valid_correspondence_threshold']
        self.icp_error_threshold: float = args['icp_error_threshold']
        self.trimming_ratio: float = args['trimming_ratio']
        self.min_distance_threshold: float = args['min_distance_threshold']
        #Initialize variables
        self.pose_segment: List[np.ndarray] = []
        self.delta_t_acc = np.eye(4)
        self.error_consecutive=0

    def align_pose(self) -> None:
        """
        Align the current sliding-window trajectory to lane centerlines.

        The method computes 2D correspondences between the XY-projected
        trajectory and nearby lanelet points (using `lane_kdtree` and
        a normal-shooting strategy), filters invalid matches, and then runs
        trimmed ICP in 2D. If the final error is below the configured
        threshold, the accumulated 2D rigid transform is applied to all
        poses in the window and stored in `self.delta_t_acc`.

        Notes
        -----
        - Uses only XY coordinates; Z and orientation are left unchanged
          by the ICP update.
        - Early exits if the traversed path length within the window is
          below `min_distance_threshold`.

        Return:
            0: trajectory length < min_distance_threshold
            1: valid correspondences < valid_correspondence_threshold
            2: ICP error < icp_error_threshold
        """
        trajectory_points_xy = np.array([[pose[0, -1], pose[1, -1]] for pose in self.pose_segment])# Xand Y points
        trajectory_distance = np.sum(np.linalg.norm(np.diff(trajectory_points_xy, axis=0), axis=1))
        
        if trajectory_distance < self.min_distance_threshold:
            return 0
            
        _,knn_index = self.lane_kdtree.query(trajectory_points_xy, k=self.knn_neighbors)
        best_lane_points = utils.find_interception_normal_shooting_nextpoint_tangent(
            trajectory_points_xy, knn_index, self.lane_points, self.lane_points_neighbour, self.lane_direction_tan
        )

        valid_mask = ~np.isnan(best_lane_points).any(axis=1)
        # return best_lane_points[valid_mask]

        if valid_mask.sum() < len(self.pose_segment) * self.valid_correspondence_threshold:
            return 1
        R_total, T_total, final_error = utils.solve_trimmed_icp_2d(
            trajectory_points_xy[valid_mask], 
            best_lane_points[valid_mask], 
            trimming_ratio=self.trimming_ratio,
        )


        if final_error < self.icp_error_threshold:
            # Apply 2D transformation to pose history
            pose_segment = []
            for pose in self.pose_segment:  
                point_xy = np.array([pose[0, -1], pose[1, -1]])
                point_xy = R_total @ point_xy + T_total
                pose[0, -1] = point_xy[0]
                pose[1, -1] = point_xy[1]  
                # pose[2, -1] remains unchanged
                pose_segment.append(pose)
            self.pose_segment = pose_segment

            #Update delta_t_acc 
            self.delta_t_acc[:2, -1] = R_total @ self.delta_t_acc[:2, -1] + T_total
            self.delta_t_acc[:2, :2] = R_total @ self.delta_t_acc[:2, :2]

            return 2
        else:
            return 3
        

    def apply(self, pose_received: np.ndarray) -> np.ndarray:
        """
        Apply the accumulated 2D correction to a new pose and update history.

        Parameters
        ----------
        pose : numpy.ndarray
            Homogeneous 4x4 pose matrix (row-major). The XY translation will
            be updated using the accumulated transform `delta_t_acc`.

        Returns
        -------
        numpy.ndarray
            The corrected 4x4 pose matrix (same object instance as the input).
        """
        pose=pose_received.copy()
        pose[:2, -1] = self.delta_t_acc[:2, :2] @ pose_received[:2, -1] + self.delta_t_acc[:2, -1]
        self.pose_segment.append(pose)
        message=5
        if len(self.pose_segment) == self.pose_segment_size:
            message=self.align_pose()
            self.pose_segment.pop(0)
            if message in  [1, 3, 4, 5]:
                self.error_consecutive+=1
            else:
                self.error_consecutive=0

            if self.error_consecutive > MAX_ERROR_CONSECUTIVE: #RESET
                self.error_consecutive=0
                self.pose_segment=[pose]
                self.delta_t_acc=np.eye(4)
                message=4 #RESET
        return self.pose_segment[-1], message



