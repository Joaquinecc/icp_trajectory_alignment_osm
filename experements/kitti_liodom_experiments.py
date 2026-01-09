# Copyright 2025 Distance Technologies Oy. For internal use only.
#
import argparse
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pykitti
from scipy.spatial.transform import Rotation
from utils_exp import compute_ape_metrics, save_poses_to_file, save_ape_to_csv

import copy
# Add workspace root to Python path so we can import osm_align from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # Go up from script/ to workspace root
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from osm_align.core.odometry_correction import OdomCorrector
from osm_align.utils import utils
from osm_align.utils.kitti_utils import angle_dict, cordinta_dict, read_kitti_pose


def precompute_sequence_data(
    seq: int,
    liodom_pose_dir: str,
    kitti_base_dir: str,
    map_path: str
) -> dict:
    """
    Pre-compute all sequence-specific data that doesn't depend on experiment parameters.
    
    Parameters
    ----------
    seq : int
        Sequence number
    liodom_pose_dir : str
        Directory containing liodom pose files
    kitti_base_dir : str
        Base directory for KITTI dataset
    map_path : str
        Base path to map points file (will be joined with sequence-specific path)
        
    Returns
    -------
    dict
        Dictionary containing pre-computed data:
        - gt_poses: Ground truth poses (transformed to velodyne frame)
        - liodom_poses: Liodom poses (transformed to velodyne frame)
        - ape_liodom: Pre-computed APE metrics for liodom poses
        - points_lane_map: Map points for this sequence
    """
    seq_str = f"{seq:02d}"
    
    # Load KITTI data
    kitti_odom = pykitti.odometry(kitti_base_dir, seq_str)
    
    # Load liodom poses
    liodom_pose_file_path = os.path.join(liodom_pose_dir, f"{seq_str}.txt")
    liodom_poses = read_kitti_pose(liodom_pose_file_path)
    
    # Transform poses
    T_cam0_velo = kitti_odom.calib.T_cam0_velo
    tf_yaw_to_enu = np.eye(4)
    tf_yaw_to_enu[:3, :3] = Rotation.from_euler('z', -angle_dict[seq_str], degrees=True).as_matrix()
    
    # Transform GT poses
    gt_poses = np.array(kitti_odom.poses) @ T_cam0_velo
    gt_poses = tf_yaw_to_enu @ np.linalg.inv(gt_poses[0]) @ gt_poses
    
    # Transform liodom poses
    liodom_poses = liodom_poses @ T_cam0_velo
    liodom_poses = tf_yaw_to_enu @ np.linalg.inv(liodom_poses[0]) @ liodom_poses
    
    # Get map points (sequence-specific map path)
    seq_map_path = os.path.join(kitti_base_dir, "map", seq_str, f"{seq_str}_map_points.npz")
    new_origin_gps = [cordinta_dict[seq_str]['origin_lat'], cordinta_dict[seq_str]['origin_lon']]
    points_lane_map = utils.get_map_points(seq_map_path, new_origin_gps)
    
    # Verify poses have same length
    assert len(liodom_poses) == len(gt_poses), \
        f"Liodom poses ({len(liodom_poses)}) and GT poses ({len(gt_poses)}) must have same length"
    
    # Pre-compute APE metrics for liodom poses
    ape_liodom = compute_ape_metrics(gt_poses, liodom_poses)
    
    return {
        'gt_poses': gt_poses,
        'liodom_poses': liodom_poses,
        'ape_liodom': ape_liodom,
        'points_lane_map': points_lane_map
    }


def run_single_experiment(
    seq: int,
    pose_segment_size: int,
    knn_neighbors: int,
    max_error_consecutive: int,
    icp_error_threshold: float,
    output_dir_results: str,
    seq_data: dict,
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
    icp_error_threshold : float
        ICP error threshold
    output_dir_results : str
        Output directory for results
    seq_data : dict
        Pre-computed sequence data containing:
        - gt_poses: Ground truth poses
        - liodom_poses: Liodom poses
        - ape_liodom: Pre-computed APE metrics for liodom
        - points_lane_map: Map points
    lock : threading.Lock
        Thread lock for printing
    """
    seq_str = f"{seq:02d}"

    try:
        # Use pre-computed sequence data (passed as copy)
        gt_poses = seq_data['gt_poses']
        liodom_poses = seq_data['liodom_poses']
        ape_liodom = seq_data['ape_liodom']  # Already a copy
        points_lane_map = seq_data['points_lane_map']

        folder_name = f"r_{seq_str}_{pose_segment_size}_{knn_neighbors}_{max_error_consecutive}_{icp_error_threshold}"
        output_folder = os.path.join(output_dir_results, folder_name)
        
        # Check if experiment already completed
        poses_path = os.path.join(output_folder, "poses.txt")
        result_liodom_path = os.path.join(output_folder, "result_liodom.csv")
        result_corrected_path = os.path.join(output_folder, "result_corrected.csv")
        
        if os.path.exists(output_folder) and \
           os.path.exists(poses_path) and \
           os.path.exists(result_liodom_path) and \
           os.path.exists(result_corrected_path):
            with lock:
                print(f"Skipping (already completed): seq={seq_str}, pose_segment_size={pose_segment_size}, "
                      f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                      f"icp_error_threshold={icp_error_threshold}")
            return
        
        os.makedirs(output_folder, exist_ok=True)
            
        # Setup correction parameters
        args = {
            'pose_segment_size': pose_segment_size,
            'knn_neighbors': knn_neighbors,
            'max_error_consecutive': max_error_consecutive,
            'valid_correspondence_threshold': 0.5,
            'trimming_ratio': 0.1,
            'min_distance_threshold': 10.0,
            'icp_error_threshold': icp_error_threshold,
        }
        
        # Create corrector
        trajectory_correction = OdomCorrector(points_lane_map, args)
        
        # Apply correction
        poses_corrected = []
        for i in range(len(liodom_poses)):
            pose_received = liodom_poses[i]
            pose_corrected, message = trajectory_correction.apply(pose_received)
            poses_corrected.append(pose_corrected)
        
        poses_corrected = np.array(poses_corrected)
        
        # Compute APE metrics (ape_liodom is already pre-computed)
        ape_corrected = compute_ape_metrics(gt_poses, poses_corrected)
        
        # Save poses
        save_poses_to_file(poses_corrected, poses_path)
        save_ape_to_csv(ape_liodom, result_liodom_path)
        save_ape_to_csv(ape_corrected, result_corrected_path)
        
        with lock:
            print(f"Completed: seq={seq_str}, pose_segment_size={pose_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                  f"icp_error_threshold={icp_error_threshold}")
            print(f"  Liodom APE RMSE: {ape_liodom['rmse']:.4f}")
            print(f"  Corrected APE RMSE: {ape_corrected['rmse']:.4f}")
    
    except Exception as e:
        with lock:
            print(f"ERROR in seq={seq_str}, pose_segment_size={pose_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                  f"icp_error_threshold={icp_error_threshold}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run liodom trajectory correction experiments')
    parser.add_argument('--liodom_pose_dir', type=str, required=True,
                        help='Directory containing liodom pose files')
    parser.add_argument('--kitti_base_dir', type=str, required=True,
                        help='Base directory for KITTI dataset')
    parser.add_argument('--map_path', type=str, required=False,
                        help='Base path to map points file (optional, will use sequence-specific paths)')
    parser.add_argument('--output_dir_results', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='Number of threads to use (default: 4)')
    
    args = parser.parse_args()
    
    # Parameter ranges
    pose_segment_sizes = [20, 50, 70, 100, 150, 200, 300]
    knn_neighbors_list = [2, 5, 10, 20, 50, 100]
    max_error_consecutive_list = [5, 10, 50, 100, 10000]
    icp_error_threshold_list = [1.0, 1.5, 2.0]
    
    # Sequences: 00-10, skipping 03
    sequences = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    
    # Create output directory
    os.makedirs(args.output_dir_results, exist_ok=True)
    
    # Pre-compute sequence data for all sequences
    print("Pre-computing sequence data...")
    sequence_data = {}
    for seq in sequences:
        seq_str = f"{seq:02d}"
        print(f"  Pre-computing data for sequence {seq_str}...")
        try:
            sequence_data[seq] = precompute_sequence_data(
                seq, args.liodom_pose_dir, args.kitti_base_dir, args.map_path
            )
        except Exception as e:
            print(f"  ERROR pre-computing data for sequence {seq_str}: {e}")
            continue
       
    print(f"Pre-computed data for {len(sequence_data)} sequences")
    
    # Generate all experiment combinations
    experiments = []
    for pose_segment_size in pose_segment_sizes:
        for knn_neighbors in knn_neighbors_list:
            for max_error_consecutive in max_error_consecutive_list:
                for icp_error_threshold in icp_error_threshold_list:
                    for seq in sequences:
                        experiments.append((
                            seq, pose_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold
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
        for seq, pose_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold in experiments:
            # Skip if sequence data wasn't pre-computed successfully
            if seq not in sequence_data:
                continue
                
            # Pass a deep copy of sequence data to each thread
            seq_data_copy = {
                'gt_poses': sequence_data[seq]['gt_poses'].copy(),
                'liodom_poses': sequence_data[seq]['liodom_poses'].copy(),
                'ape_liodom': copy.deepcopy(sequence_data[seq]['ape_liodom']),  # Dictionary needs deep copy
                'points_lane_map': sequence_data[seq]['points_lane_map']  # Map points can be shared
            }
            future = executor.submit(
                run_single_experiment,
                seq, pose_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold,
                args.output_dir_results, seq_data_copy, lock
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

