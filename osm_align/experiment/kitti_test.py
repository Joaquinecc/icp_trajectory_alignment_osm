import os

from osm_align.utils.kitti_utils import angle_dict, cordinta_dict, get_pose
import numpy as np
from scipy.spatial import cKDTree
import lanelet2
from ..odometry_correction import OdomCorrector
from evo.main_ape import ape
from evo.main_rpe import rpe
from evo.core.trajectory import PosePath3D, Plane
from evo.core import metrics
from evo.core.units import Unit
import copy
import time
from typing import List


class Odomcorrection():
    
    def __init__(
        self,
        frame_id,
        pose_segment_size,
        icp_error_threshold,
        trimming_ratio,
        knn_neighbors,
        valid_correspondence_threshold,
        min_distance_threshold,
    ) -> None:
        self.frame_id = frame_id
        self.min_distance_threshold = min_distance_threshold
        self.pose_segment_size = pose_segment_size
        self.icp_error_threshold = icp_error_threshold
        self.trimming_ratio = trimming_ratio
        self.knn_neighbors = knn_neighbors
        self.valid_correspondence_threshold = valid_correspondence_threshold
        self.frame_counter = 0
        self.pose_history = []
        self.pose_segment = []
        
        self.map_lanelet_path: str = f'/home/joaquinecc/Documents/dataset/kitti/dataset/map/{self.frame_id}/lanelet2_seq_{self.frame_id}.osm'

        self.origin_coords_lanelet: List[float] = [cordinta_dict[self.frame_id]['origin_lat'], cordinta_dict[self.frame_id]['origin_lon']]
        self.angle_lanelet_correction: float = angle_dict[self.frame_id]
        
        # Setup save paths and create folder
        self.folder = f"{self.pose_segment_size}_{self.icp_error_threshold}_{self.trimming_ratio}_{self.knn_neighbors}_{self.valid_correspondence_threshold}"
        self.save_path_folder = f"/home/joaquinecc/Documents/ros_projects/results/odom_correction/{self.frame_id}/{self.folder}"
        os.makedirs(self.save_path_folder, exist_ok=True)
        
        # Setup logging to capture all prints
        self.log_file = os.path.join(self.save_path_folder, "log.txt")
        self.log_handle = open(self.log_file, 'w')
        
        # Print all parameters for debugging/logging
        self._log_info(
            f"Parameters:\n"
            f"  frame_id: {self.frame_id}\n"
            f"  map_lanelet_path: {self.map_lanelet_path}\n"
            f"  pose_segment_size: {self.pose_segment_size}\n"
            f"  knn_neighbors: {self.knn_neighbors}\n"
            f"  valid_correspondence_threshold: {self.valid_correspondence_threshold}\n"
            f"  icp_error_threshold: {self.icp_error_threshold}\n"
            f"  trimming_ratio: {self.trimming_ratio}\n"
            f"  min_distance_threshold: {self.min_distance_threshold}\n"
            f"  save_path_folder: {self.save_path_folder}"
        )

        # Load lanelet map and build KD-tree
        self._load_lanelet_map()
        self._build_lane_kdtree()
        
        # Initialize the generic odometry corrector
        self.corrector = OdomCorrector(
            lane_points=self.lane_points,
            lane_points_next=self.lane_points_next,
            lane_kdtree=self.lane_kdtree,
            args={
                'pose_segment_size': self.pose_segment_size,
                'knn_neighbors': self.knn_neighbors,
                'valid_correspondence_threshold': self.valid_correspondence_threshold,
                'icp_error_threshold': self.icp_error_threshold,
                'trimming_ratio': self.trimming_ratio,
                'min_distance_threshold': self.min_distance_threshold,
            }
        )
        
        self.liodom_path_pose = f"/home/joaquinecc/Documents/ros_projects/results/liodom/{frame_id}/poses.txt"
        self.liodom_poses = get_pose(self.liodom_path_pose) #Velodyne frame
        
        start_time = time.time()
        self.run_process()
        elapsed_time = time.time() - start_time
        self._log_info(f"run_process elapsed time: {elapsed_time:.3f} s")
        
        self.calculate_ape_rpe()
        self._close_log()

    def _log_info(self, message):
        """Log message to both console and file"""
        # print(message)
        self.log_handle.write(message + '\n')
        self.log_handle.flush()

    def _close_log(self):
        """Close the log file"""
        if hasattr(self, 'log_handle') and self.log_handle:
            self.log_handle.close()

    def calculate_ape_rpe(self):
        gt_path_pose = f'/home/joaquinecc/Documents/dataset/kitti/dataset/poses/{self.frame_id}.txt'
        tf_base_to_velo = np.eye(4)
        tf_base_to_velo[:3, :3] = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        gt_poses = T_kitti @ get_pose(gt_path_pose) #Velodyne frame
        traj_GT = PosePath3D(poses_se3=gt_poses)
        traj_est = PosePath3D(poses_se3=self.pose_history)
        
        # APE calculation
        ape_metric = ape(
            traj_ref=copy.deepcopy(traj_GT),
            traj_est=copy.deepcopy(traj_est),
            pose_relation=metrics.PoseRelation.translation_part,
            project_to_plane=Plane.XY
        )
        ape_stat = ape_metric.stats
        filtered_ape_stats = {
            'rmse': float(ape_stat['rmse']),
            'mean': float(ape_stat['mean']),
            'median': float(ape_stat['median'])
        }

        # RPE calculation
        rpe_metric = rpe(
            traj_ref=copy.deepcopy(traj_GT),
            traj_est=copy.deepcopy(traj_est),
            pose_relation=metrics.PoseRelation.translation_part,
            project_to_plane=Plane.XY,
            delta=100,
            delta_unit=Unit.meters,
            all_pairs=True
        )
        rpe_stat = rpe_metric.stats
        filtered_rpe_stats = {
            'rmse': float(rpe_stat['rmse']),
            'mean': float(rpe_stat['mean']),
            'median': float(rpe_stat['median'])
        }
        
        # Log and save results
        self._log_info(f"\n=== Results for Frame {self.frame_id} ===")
        self._log_info(f"APE Stats: RMSE={filtered_ape_stats['rmse']:.4f}, Mean={filtered_ape_stats['mean']:.4f}, Median={filtered_ape_stats['median']:.4f}")
        self._log_info(f"RPE Stats: RMSE={filtered_rpe_stats['rmse']:.4f}, Mean={filtered_rpe_stats['mean']:.4f}, Median={filtered_rpe_stats['median']:.4f}")
        

        print(f"\n=== Results for Frame {self.frame_id} ===")
        print(f"APE Stats: RMSE={filtered_ape_stats['rmse']:.4f}, Mean={filtered_ape_stats['mean']:.4f}, Median={filtered_ape_stats['median']:.4f}")
        print(f"RPE Stats: RMSE={filtered_rpe_stats['rmse']:.4f}, Mean={filtered_rpe_stats['mean']:.4f}, Median={filtered_rpe_stats['median']:.4f}")
        
        # Save results to file
        results_file = os.path.join(self.save_path_folder, "results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Results for Frame {self.frame_id}\n")
            f.write(f"Parameters: {self.folder}\n\n")
            f.write(f"APE Stats:\n")
            f.write(f"  RMSE: {filtered_ape_stats['rmse']:.6f}\n")
            f.write(f"  Mean: {filtered_ape_stats['mean']:.6f}\n")
            f.write(f"  Median: {filtered_ape_stats['median']:.6f}\n\n")
            f.write(f"RPE Stats:\n")
            f.write(f"  RMSE: {filtered_rpe_stats['rmse']:.6f}\n")
            f.write(f"  Mean: {filtered_rpe_stats['mean']:.6f}\n")
            f.write(f"  Median: {filtered_rpe_stats['median']:.6f}\n")
        
        self._log_info(f"Results saved to: {results_file}")

    def run_process(self):
        for liodom_pose in self.liodom_poses:
            pose = liodom_pose.copy()
            # Apply accumulated correction via OdomCorrector and update window
            pose = self.corrector.apply(pose)
            self.pose_segment.append(pose)
            
            self.pose_history.append(self.pose_segment[-1].copy())
            self.frame_counter += 1
            
        self.pose_history = np.array(self.pose_history)
        
        # Save pose history
        poses_file = os.path.join(self.save_path_folder, "poses.txt")
        # Save as 12 values per line (3x4 row-major), as in KITTI odometry format
        with open(poses_file, "w") as f:
            for pose in self.pose_history:
                # pose is 4x4, take first 3 rows, all 4 columns, flatten row-major
                vals = pose[:3, :].reshape(-1)
                f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")
        self._log_info(f"Pose history saved to: {poses_file}")
        self._log_info(f"Total frames processed: {self.frame_counter}")
        self._log_info(f"Total poses in history: {len(self.pose_history)}")

    def _load_lanelet_map(self) -> None:
        """
        Load lanelet map from OpenStreetMap file using UTM projection.

        Initializes the UTM projector with the provided origin coordinates
        and loads the lanelet map data for subsequent processing.

        Raises
        ------
        Exception
        If the lanelet map file cannot be loaded or is corrupted.

        Notes
        -----
        The projector converts GPS coordinates to local UTM coordinates
        for efficient geometric computations during trajectory alignment.
        """
        self.projector = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(self.origin_coords_lanelet[0], self.origin_coords_lanelet[1])
        )
        self.lanelet_map = lanelet2.io.load(self.map_lanelet_path, self.projector)
        
        self._log_info(f"Lanelet map loaded from: {self.map_lanelet_path}")
        self._log_info(f"Origin coordinates: {self.origin_coords_lanelet}")
        self._log_info(f"Angle correction: {self.angle_lanelet_correction} degrees")

    def _build_lane_kdtree(self) -> None:
        """
        Extract lanelet centerlines and build spatial index for efficient queries.

        Processes all lanelets in the map to extract centerline points, applies
        coordinate system rotation, removes duplicate points, and builds a KD-tree
        for fast nearest neighbor searches during trajectory alignment.

        The algorithm:
        1. Iterates through all lanelets and extracts centerline points
        2. Applies rotation matrix to align with vehicle coordinate system
        3. Filters points to maintain minimum spacing (1.0 meter)
        4. Creates "next point" associations for tangent vector computation
        5. Builds KD-tree spatial index

        Notes
        -----
        The minimum distance filtering (min_dist=1.0) reduces computational
        complexity while preserving map geometry for alignment purposes.
        """
        # Define rotation matrix
        rotation_angle = np.radians(self.angle_lanelet_correction)
        R_M = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        lane_points = []
        min_dist = 3.0
        lane_points_next = []
        
        for lanelet in self.lanelet_map.laneletLayer:
            prev_point = None
            centerline = lanelet.centerline
            aux_points = []
            for i, point in enumerate(centerline):
                corrected_point = R_M @ np.array([point.x, point.y])
                if prev_point is not None:
                    if np.linalg.norm(corrected_point - prev_point) < min_dist:
                        continue
                prev_point = corrected_point
                aux_points.append(corrected_point)
            lane_points.extend(aux_points)
            
            # Create next-point associations for tangent computation
            for i in range(len(aux_points)):
                nn = None
                if i < len(aux_points) - 1:
                    nn = aux_points[i + 1]
                else:
                    nn = aux_points[i - 1]
                lane_points_next.append(nn)

        self.lane_points_next = np.array(lane_points_next)
        self.lane_points = np.array(lane_points)
        self.lane_kdtree = cKDTree(self.lane_points)
        
        self._log_info(f"Built KD-tree with {len(lane_points)} lane points")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Odomcorrection with specified parameters.")
    parser.add_argument("--frame_id", type=str, default="00", help="Frame ID")
    parser.add_argument("--pose_segment_size", type=int, default=10, help="Pose segment size")
    parser.add_argument("--icp_error_threshold", type=float, default=0.5, help="ICP error threshold")
    parser.add_argument("--trimming_ratio", type=float, default=0.8, help="Trimming ratio")
    parser.add_argument("--knn_neighbors", type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument("--valid_correspondence_threshold", type=float, default=0.6, help="Valid correspondence threshold")
    parser.add_argument("--min_distance_threshold", type=float, default=10.0, help="Minimum distance threshold")
    args = parser.parse_args()



    corrector = Odomcorrection(
        frame_id=args.frame_id,
        pose_segment_size=args.pose_segment_size,
        icp_error_threshold=args.icp_error_threshold,
        trimming_ratio=args.trimming_ratio,
        knn_neighbors=args.knn_neighbors,
        valid_correspondence_threshold=args.valid_correspondence_threshold,
        min_distance_threshold=args.min_distance_threshold
    )

# python -m osm_align.experiment.kitti_test --frame_id 00 --pose_segment_size 20 --icp_error_threshold 1.0 --trimming_ratio 0.4 --knn_neighbors 10 --valid_correspondence_threshold 0.8 