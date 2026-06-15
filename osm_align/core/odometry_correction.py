"""Odometry correction module using 2D trimmed ICP against lanelet centerlines."""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

# from geometry_msgs.msg import Pose  # Not used here; keep minimal deps
try:
    import osm_align.utils.utils as utils
except ImportError:
    import utils.utils as utils


# Configuration parameters are now declared as ROS parameters in the node
class OdomCorrector:
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
        - 'min_segment_size' (int): size of the sliding window
        - 'knn_neighbors' (int): number of neighbors for correspondence
        - 'valid_correspondence_threshold' (float): min fraction of valid
          correspondences required to run ICP
        - 'icp_error_threshold' (float): maximum ICP error to accept update
        - 'trimming_ratio' (float): fraction of largest residuals to trim
        - 'min_distance_threshold' (float): minimum path length in the
          current window to trigger alignment
        - 'max_error_consecutive' (int, optional): maximum consecutive errors
          before resetting the correction. Defaults to 1000 if not provided.
    """

    def __init__(
        self,
        points_lane_map,  # is a 6xN array of points
        args: Dict[str, Any],
    ) -> None:

        self.lane_points = points_lane_map[:, 0:2]
        self.lane_points_neighbour = points_lane_map[:, 2:4]  # To get intersections
        self.lane_direction_tan = points_lane_map[:, 4:6]  # Car direction

        self.lane_kdtree: Optional[cKDTree] = cKDTree(self.lane_points)

        self.min_segment_size: int = args["min_segment_size"]
        self.knn_neighbors: int = args["knn_neighbors"]
        self.valid_correspondence_threshold: float = args.get(
            "valid_correspondence_threshold", 0.5
        )
        self.icp_error_threshold: float = args["icp_error_threshold"]
        self.trimming_ratio: float = args.get("trimming_ratio", 0.1)
        self.min_distance_threshold: float = args.get("min_distance_threshold", 3.0)
        self.max_error_consecutive: int = args.get("max_error_consecutive", 1000)
        self.max_segment_size: int = args.get("max_segment_size", 1000)
        # Initialize variables
        self.poses_corrected: np.ndarray = np.array([])
        self.lane_points_matched: np.ndarray = np.array([])
        self.poses_original = deque(maxlen=self.min_segment_size)
        self.delta_t_acc = np.eye(4)
        self.error_counter = 0

        self.dynamic_segment_size = self.min_segment_size
        # Cache for best lane point matches: list of dicts, one per pose
        # Each dict maps (pointx, pointy) tuple to counter
        self.lane_point_cache: List[Dict[tuple, int]] = []
        # self._set_messages_info()

    def align_trajectory_pose(self) -> None:
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
        # trajectory_points_xy = np.array([[pose[0, -1], pose[1, -1]] for pose in self.poses_corrected])# Xand Y points
        trajectory_distance = np.sum(
            np.linalg.norm(np.diff(self.poses_corrected, axis=0), axis=1)
        )

        if trajectory_distance < self.min_distance_threshold:
            return 0

        trajectory_points_xy = np.array(self.poses_corrected)

        _, knn_index = self.lane_kdtree.query(
            trajectory_points_xy, k=self.knn_neighbors
        )
        best_lane_points = self.find_best_match_lane_point(
            trajectory_points_xy, knn_index
        )

        # self.lane_points_matched = np.vstack([self.lane_points_matched, best_lane_points[0]])

        valid_mask = ~np.isnan(best_lane_points).any(axis=1)
        # return best_lane_points[valid_mask]

        if (
            valid_mask.sum()
            < len(trajectory_points_xy) * self.valid_correspondence_threshold
        ):
            return 1
        R_total, T_total, final_error = utils.solve_trimmed_icp_2d(
            trajectory_points_xy[valid_mask],
            best_lane_points[valid_mask],
            trimming_ratio=self.trimming_ratio,
        )

        if final_error < self.icp_error_threshold:
            self.poses_corrected = (R_total @ self.poses_corrected.T).T + T_total

            # Update delta_t_acc
            self.delta_t_acc[:2, -1] = R_total @ self.delta_t_acc[:2, -1] + T_total
            self.delta_t_acc[:2, :2] = R_total @ self.delta_t_acc[:2, :2]

            return 2
        else:
            return 3

    def find_best_match_lane_point(
        self,
        trajectory_points: np.ndarray,
        nearest_lane_points_index: List[List[int]],
    ) -> np.ndarray:
        intercept_points = np.full_like(trajectory_points, np.nan)

        for i, p in enumerate(trajectory_points):
            # Check cache first - if we have a cached match with counter >= 4, skip search
            if i < len(self.lane_point_cache):
                cache_dict = self.lane_point_cache[i]
                # Find cached match with counter >= 4
                for (cached_x, cached_y), counter in cache_dict.items():
                    if counter >= 4:
                        intercept_points[i] = np.array([cached_x, cached_y])
                        # Increment counter for this cached match
                        cache_dict[(cached_x, cached_y)] = counter + 1
                        break
                # If we found a cached match, skip the search
                if not np.isnan(intercept_points[i]).any():
                    continue

            # Estimate tangent direction from trajectory
            if 0 < i < len(trajectory_points) - 1:  # Middle point
                tangent_traj = trajectory_points[i + 1] - trajectory_points[i - 1]
            elif i == len(trajectory_points) - 1:  # Last point
                tangent_traj = p - trajectory_points[i - 1]
            elif i == 0:  # First point
                tangent_traj = trajectory_points[i + 1] - p
            tangent_traj = tangent_traj / np.linalg.norm(tangent_traj)
            normal_traj = np.array(
                [tangent_traj[1], -tangent_traj[0]]
            )  # Perpendicular to tangent

            best_intercept_point = np.nan
            nearest_lane_point_idx = nearest_lane_points_index[i]

            for laned_idx in nearest_lane_point_idx:
                parallel_score = np.dot(
                    tangent_traj, self.lane_direction_tan[laned_idx]
                )
                if parallel_score < 0.95:  # Similar road direction to pose yaw
                    continue
                lane_point = self.lane_points[laned_idx]
                neighbour_point = self.lane_points_neighbour[laned_idx]
                # Find intersection of normal_traj at p with the segment ab
                # Solve: a + t * ab = p + s * normal_traj
                # => t * ab - s * normal_traj = (p - a)
                ab = neighbour_point - lane_point
                A = np.column_stack((ab, -normal_traj))
                det = np.linalg.det(A)
                if abs(det) > 1e-10:
                    sol = np.linalg.inv(A) @ (p - lane_point)
                    t = sol[0]
                    # Only accept intersection if t in [0,1] (segment)
                    if 0.0 <= t <= 1.0:  # There is an intersection
                        proj = lane_point + t * ab
                        best_intercept_point = proj
                        break

            intercept_points[i] = best_intercept_point

            # Update cache with the found match
            if i >= len(self.lane_point_cache):
                # Extend cache if needed (shouldn't happen, but safety check)
                while len(self.lane_point_cache) <= i:
                    self.lane_point_cache.append({})

            if not np.isnan(best_intercept_point).any():
                # Convert to tuple for dictionary key
                match_key = (
                    float(best_intercept_point[0]),
                    float(best_intercept_point[1]),
                )
                # Update or initialize counter
                if i < len(self.lane_point_cache):
                    if match_key in self.lane_point_cache[i]:
                        # Same match found, increment counter
                        self.lane_point_cache[i][match_key] += 1
                    else:
                        # Different match found, replace old entry
                        self.lane_point_cache[i] = {match_key: 1}
        return intercept_points

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
        # pose[:2, -1] = self.delta_t_acc[:2, :2] @ pose_received[:2, -1] + self.delta_t_acc[:2, -1]
        pose_xy = (
            self.delta_t_acc[:2, :2] @ pose_received[:2, -1] + self.delta_t_acc[:2, -1]
        )
        if self.poses_corrected.size:
            self.poses_corrected = np.vstack([self.poses_corrected, pose_xy])
        else:
            self.poses_corrected = np.expand_dims(pose_xy, axis=0)
        # Add empty cache entry for new pose
        self.lane_point_cache.append({})
        self.poses_original.append(pose_xy)

        # Enforce max_segment_size constraint: drop oldest 20 poses if exceeded
        if len(self.poses_corrected) > self.max_segment_size:
            self.poses_corrected = self.poses_corrected[20:]  # Remove oldest 20 poses
            if len(self.lane_point_cache) >= 20:
                self.lane_point_cache = self.lane_point_cache[
                    20:
                ]  # Remove corresponding cache entries

        message = 5
        if len(self.poses_corrected) == self.dynamic_segment_size:
            message = self.align_trajectory_pose()
            if message in [1, 3, 4]:
                self.error_counter += 1
                self.poses_corrected = self.poses_corrected[1:]  # pop the oldest pose
                # Remove first cache entry to keep cache synchronized
                if len(self.lane_point_cache) > 0:
                    self.lane_point_cache.pop(0)

            else:  # no error
                self.error_counter = 0
                # Cap dynamic_segment_size at max_segment_size
                if self.dynamic_segment_size < self.max_segment_size:
                    self.dynamic_segment_size += 1
                # self.poses_corrected=self.poses_corrected[1:]#pop the last point

            if self.error_counter > self.max_error_consecutive:  # RESET
                self.error_counter = 0
                # self.poses_corrected=[pose]
                self.poses_corrected = self.poses_corrected[
                    -self.min_segment_size + 1 :
                ]  # keep the last min_segment_size-1 points
                # Synchronize cache: keep only the last min_segment_size-1 entries
                self.lane_point_cache = self.lane_point_cache[
                    -(self.min_segment_size - 1) :
                ]
                # self.poses_corrected=np.array(self.poses_original)[1:]
                self.dynamic_segment_size = self.min_segment_size
                assert len(self.poses_corrected) == self.min_segment_size - 1
                assert len(self.lane_point_cache) == len(self.poses_corrected)
                # self.delta_t_acc=np.eye(4)
                message = 4  # RESET
        pose = pose_received.copy()
        pose[:2, -1] = self.poses_corrected[-1]
        return pose, message

    def _get_messages_str(self, i: int) -> None:
        """
        Set the messages information for the trajectory correction.
        The messages are used to print the information of the trajectory correction.
        """
        messages_info = {
            0: f"trajectory length < {self.min_distance_threshold}, skip ICP",
            1: f"valid correspondences < {self.valid_correspondence_threshold}, skip ICP",
            2: f"ICP error < {self.icp_error_threshold}, ICP success with {self.dynamic_segment_size} points",
            3: f"ICP error > {self.icp_error_threshold}, ICP failed",
            4: "RESET",
            5: "Not enought points to align",
            6: "Not initialized",
        }
        return messages_info[i]

    def get_message_str(self, message_code: int) -> str:
        return self._get_messages_str(message_code)
