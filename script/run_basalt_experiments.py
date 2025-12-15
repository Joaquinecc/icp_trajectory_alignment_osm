# Copyright 2025 Distance Technologies Oy. For internal use only.
#
import argparse
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple
import csv
import pykitti
from scipy.spatial.transform import Rotation
from evo.main_ape import ape
from evo.core.trajectory import PosePath3D, Plane
from evo.core import metrics
import copy

# Add workspace root to Python path so we can import osm_align from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # Go up from script/ to workspace root
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from osm_align.core.odometry_correction import OdomCorrector
from osm_align.utils import utils
from osm_align.utils.kitti_utils import angle_dict, cordinta_dict


def compute_ape_metrics(gt_poses: np.ndarray, est_poses: np.ndarray) -> Dict:
    """
    Compute APE (Absolute Pose Error) metrics between ground truth and estimated poses.
    
    Parameters
    ----------
    gt_poses : np.ndarray
        Ground truth poses as (N, 4, 4) array
    est_poses : np.ndarray
        Estimated poses as (N, 4, 4) array
    
    Returns
    -------
    Dict
        Dictionary containing APE metrics: rmse, mean, median, std, min, max, sse
    """
    traj_GT = PosePath3D(poses_se3=gt_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    
    ape_metric = ape(
        traj_ref=copy.deepcopy(traj_GT),
        traj_est=copy.deepcopy(traj_est),
        pose_relation=metrics.PoseRelation.translation_part,
        project_to_plane=Plane.XY
    )
    
    ape_stat = ape_metric.stats
    
    return {
        'rmse': float(ape_stat['rmse']),
        'mean': float(ape_stat['mean']),
        'median': float(ape_stat['median']),
        'std': float(ape_stat['std']),
        'min': float(ape_stat['min']),
        'max': float(ape_stat['max']),
        'sse': float(ape_stat['sse'])
    }


def save_poses_to_file(poses: np.ndarray, filepath: str):
    """
    Save poses to file in KITTI format (12 values per line: 3x4 row-major).
    
    Parameters
    ----------
    poses : np.ndarray
        Array of (N, 4, 4) poses
    filepath : str
        Path to output file
    """
    with open(filepath, 'w') as f:
        for pose in poses:
            vals = pose[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")


def save_ape_to_csv(ape_metrics: Dict, filepath: str):
    """
    Save APE metrics to CSV file.
    
    Parameters
    ----------
    ape_metrics : Dict
        Dictionary containing APE metrics
    filepath : str
        Path to output CSV file
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in ape_metrics.items():
            writer.writerow([key, value])


def run_single_experiment(
    seq: int,
    pose_segment_size: int,
    knn_neighbors: int,
    max_error_consecutive: int,
    basalt_pose_dir: str,
    kitti_base_dir: str,
    map_path: str,
    output_dir_results: str,
    lock: threading.Lock
):
    """
    Run a single experiment for a given sequence and parameter combination.
    
    Parameters
    ----------
    seq : int
        Sequence number (0-10, skipping 3)
    pose_segment_size : int
        Size of pose segment window
    knn_neighbors : int
        Number of KNN neighbors
    max_error_consecutive : int
        Maximum consecutive errors before reset
    basalt_pose_dir : str
        Directory containing basalt pose CSV files
    kitti_base_dir : str
        Base directory for KITTI dataset
    map_path : str
        Path to map points file
    output_dir_results : str
        Output directory for results
    lock : threading.Lock
        Thread lock for printing
    """
    seq_str = f"{seq:02d}"
    
    try:
        # Load KITTI data
        kitti_odom = pykitti.odometry(kitti_base_dir, seq_str)
        
        # Load basalt poses
        basalt_pose_file_path = os.path.join(basalt_pose_dir, f"{seq_str}.csv")
        basalt_poses = utils.read_basalt_pose(basalt_pose_file_path)
        
        # Transform poses
        T_cam0_velo = kitti_odom.calib.T_cam0_velo
        tf_yaw_to_enu = np.eye(4)
        tf_yaw_to_enu[:3, :3] = Rotation.from_euler('z', -angle_dict[seq_str], degrees=True).as_matrix()
        
        # Transform GT poses
        gt_poses = np.array(kitti_odom.poses) @ T_cam0_velo
        gt_poses = tf_yaw_to_enu @ np.linalg.inv(gt_poses[0]) @ gt_poses
        
        # Transform basalt poses
        basalt_poses = basalt_poses @ T_cam0_velo
        basalt_poses = tf_yaw_to_enu @ np.linalg.inv(basalt_poses[0]) @ basalt_poses
        
        # Get map points
        new_origin_gps = [cordinta_dict[seq_str]['origin_lat'], cordinta_dict[seq_str]['origin_lon']]
        points_lane_map = utils.get_map_points(map_path, new_origin_gps)
        
        # Setup correction parameters
        args = {
            'pose_segment_size': pose_segment_size,
            'knn_neighbors': knn_neighbors,
            'max_error_consecutive': max_error_consecutive,
            'valid_correspondence_threshold': 0.5,
            'trimming_ratio': 0.1,
            'min_distance_threshold': 10.0,
            'icp_error_threshold': 1.0,
        }
        
        # Create corrector
        trajectory_correction = OdomCorrector(points_lane_map, args)
        
        # Apply correction
        poses_corrected = []
        for i in range(len(basalt_poses)):
            pose_received = basalt_poses[i]
            pose_corrected, message = trajectory_correction.apply(pose_received)
            poses_corrected.append(pose_corrected)
        
        poses_corrected = np.array(poses_corrected)
        
        # Compute APE metrics
        ape_basalt = compute_ape_metrics(gt_poses, basalt_poses)
        ape_corrected = compute_ape_metrics(gt_poses, poses_corrected)
        
        # Create output directory
        folder_name = f"r_{seq_str}_{pose_segment_size}_{knn_neighbors}_{max_error_consecutive}"
        output_folder = os.path.join(output_dir_results, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Save poses
        poses_path = os.path.join(output_folder, "poses.txt")
        # Save APE metrics
        result_basalt_path = os.path.join(output_folder, "result_basalt.csv")
        result_corrected_path = os.path.join(output_folder, "result_corrected.csv")

        save_poses_to_file(poses_corrected, poses_path)
        save_ape_to_csv(ape_basalt, result_basalt_path)
        save_ape_to_csv(ape_corrected, result_corrected_path)
        
        with lock:
            print(f"Completed: seq={seq_str}, pose_segment_size={pose_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}")
            print(f"  Basalt APE RMSE: {ape_basalt['rmse']:.4f}")
            print(f"  Corrected APE RMSE: {ape_corrected['rmse']:.4f}")
    
    except Exception as e:
        with lock:
            print(f"ERROR in seq={seq_str}, pose_segment_size={pose_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run basalt trajectory correction experiments')
    parser.add_argument('--basalt_pose_dir', type=str, required=True,
                        help='Directory containing basalt pose CSV files')
    parser.add_argument('--kitti_base_dir', type=str, required=True,
                        help='Base directory for KITTI dataset')
    parser.add_argument('--map_path', type=str, required=True,
                        help='Path to map points file (.npz)')
    parser.add_argument('--output_dir_results', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='Number of threads to use (default: 4)')
    
    args = parser.parse_args()
    
    # Parameter ranges
    pose_segment_sizes = [20, 50, 70, 100, 150, 200, 300]
    knn_neighbors_list = [2, 5, 10, 20, 50, 100]
    max_error_consecutive_list = [5, 10, 50, 100, 10000]
    
    # Sequences: 00-10, skipping 03
    sequences = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    
    # Create output directory
    os.makedirs(args.output_dir_results, exist_ok=True)
    
    # Generate all experiment combinations
    experiments = []
    for seq in sequences:
        for pose_segment_size in pose_segment_sizes:
            for knn_neighbors in knn_neighbors_list:
                for max_error_consecutive in max_error_consecutive_list:
                    experiments.append((
                        seq, pose_segment_size, knn_neighbors, max_error_consecutive
                    ))
    
    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")
    print(f"Using {args.n_threads} threads")
    print(f"Output directory: {args.output_dir_results}")
    
    # Thread lock for printing
    lock = threading.Lock()
    
    # Run experiments with thread pool
    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = []
        for seq, pose_segment_size, knn_neighbors, max_error_consecutive in experiments:
            future = executor.submit(
                run_single_experiment,
                seq, pose_segment_size, knn_neighbors, max_error_consecutive,
                args.basalt_pose_dir, args.kitti_base_dir, args.map_path,
                args.output_dir_results, lock
            )
            futures.append(future)
        
        # Wait for all experiments to complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            with lock:
                print(f"Progress: {completed}/{total_experiments} experiments completed")
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                with lock:
                    print(f"Experiment failed with exception: {e}")
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()

