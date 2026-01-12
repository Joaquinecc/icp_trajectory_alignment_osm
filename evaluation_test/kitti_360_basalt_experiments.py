# Copyright 2025 Distance Technologies Oy. For internal use only.
#
import argparse
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from utils_exp import compute_ape_metrics, save_poses_to_file, save_ape_to_csv

import copy
# Add workspace root to Python path so we can import osm_align from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # Go up from script/ to workspace root
print(workspace_root)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from osm_align.core.odometry_correction import OdomCorrector
from osm_align.utils import utils
from osm_align.utils.kitti_utils import cordinta_dict_360, read_kitti_360_poses,read_calib_file_kitti_360,kitti_360_frame_range_cam


def precompute_sequence_data(
    seq: int,
    basalt_pose_dir: str,
    kitti_360_poses: str,
    map_path: str,
    tf_cam_to_velo: np.ndarray
) -> dict:
    """
    Pre-compute all sequence-specific data that doesn't depend on experiment parameters.
    
    Parameters
    ----------
    seq : int
        Sequence number
    basalt_pose_dir : str
        Directory containing basalt pose CSV files
    kitti_360_poses : str
        Base directory for KITTI 360 poses
    map_path : str
        Path to map points file
    tf_cam_to_velo : np.ndarray
        Transformation matrix from camera to velodyne
        
    Returns
    -------
    dict
        Dictionary containing pre-computed data:
        - gt_poses: Ground truth poses (transformed to velodyne frame)
        - basalt_poses: Basalt poses (transformed to velodyne frame)
        - valid_index: Valid index array for alignment
        - ape_basalt: Pre-computed APE metrics for basalt poses
        - points_lane_map: Map points for this sequence
    """
    seq_str = f"{seq:02d}"
    
    # Get GPS origin for this sequence
    new_origin_gps = [cordinta_dict_360[seq_str]['origin_lat'], cordinta_dict_360[seq_str]['origin_lon']]
    points_lane_map = utils.get_map_points(map_path, new_origin_gps)
    
    # Load poses
    gt_poses, gt_frame_id = read_kitti_360_poses(
        os.path.join(kitti_360_poses, f"2013_05_28_drive_00{seq_str}_sync/cam0_to_world.txt")
    )
    basalt_poses = utils.read_basalt_pose(os.path.join(basalt_pose_dir, f"{seq_str}.csv"))
    
    # Align gt poses with basalt poses
    basal_frame_start, basal_frame_end = kitti_360_frame_range_cam[seq_str]
    start_frame = max(basal_frame_start, gt_frame_id[0])
    end_frame = min(basal_frame_end, gt_frame_id[-1])
    
    gt_start_frame = np.where(gt_frame_id == start_frame)[0][0]
    gt_end_frame = np.where(gt_frame_id == end_frame)[0][0]
    
    gt_poses = gt_poses[gt_start_frame:gt_end_frame+1]
    gt_poses[:, :3, -1] -= gt_poses[0, :3, -1]  # Center to 0
    
    valid_index = gt_frame_id[gt_start_frame:gt_end_frame+1] - basal_frame_start
    
    assert basalt_poses[valid_index].shape[0] == gt_poses.shape[0]
    
    # Tf to Enu
    tf_enu = np.eye(4)
    i = 1 if seq == 4 else 0
    tf_enu[:3, :3] = gt_poses[i, :3, :3].copy()
    basalt_poses = tf_enu @ basalt_poses
    
    # Transform to Velodyne frame
    gt_poses = gt_poses @ tf_cam_to_velo
    basalt_poses = basalt_poses @ tf_cam_to_velo
    
    # Pre-compute APE metrics for basalt poses
    ape_basalt = compute_ape_metrics(gt_poses, basalt_poses[valid_index])
    
    return {
        'gt_poses': gt_poses,
        'basalt_poses': basalt_poses,
        'valid_index': valid_index,
        'ape_basalt': ape_basalt,
        'points_lane_map': points_lane_map
    }


def run_single_experiment(
    seq: int,
    min_segment_size: int,
    knn_neighbors: int,
    max_error_consecutive: int,
    icp_error_threshold: float,
    output_dir_results: str,
    seq_data: dict,
    lock: multiprocessing.Lock
):
    """
    Run a single experiment for a given sequence and parameter combination.
    
    Parameters
    ----------
    seq : int
        Sequence number (0-10, skipping 3)
    min_segment_size : int
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
        - basalt_poses: Basalt poses
        - valid_index: Valid index array
        - ape_basalt: Pre-computed APE metrics for basalt
        - points_lane_map: Map points
    lock : multiprocessing.Lock
        Process lock for synchronized printing
    """
    seq_str = f"{seq:02d}"

    try:
        # Start timing
        start_time = time.time()
        
        # Use pre-computed sequence data (passed as copy)
        gt_poses = seq_data['gt_poses']
        basalt_poses = seq_data['basalt_poses']
        valid_index = seq_data['valid_index']
        ape_basalt = seq_data['ape_basalt']  # Already a copy
        points_lane_map = seq_data['points_lane_map']

        folder_name = f"r_{seq_str}_{min_segment_size}_{knn_neighbors}_{max_error_consecutive}_{icp_error_threshold}"
        output_folder = os.path.join(output_dir_results, folder_name)
        
        # Check if experiment already completed
        poses_path = os.path.join(output_folder, "poses.txt")
        result_basalt_path = os.path.join(output_folder, "result_basalt.csv")
        result_corrected_path = os.path.join(output_folder, "result_corrected.csv")
        
        if os.path.exists(output_folder) and \
           os.path.exists(poses_path) and \
           os.path.exists(result_basalt_path) and \
           os.path.exists(result_corrected_path):
            with lock:
                print(f"Skipping (already completed): seq={seq_str}, min_segment_size={min_segment_size}, "
                      f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                      f"icp_error_threshold={icp_error_threshold}")
            return
        
        os.makedirs(output_folder, exist_ok=True)
            
        # Setup correction parameters
        args = {
            'min_segment_size': min_segment_size,
            'knn_neighbors': knn_neighbors,
            'max_error_consecutive': max_error_consecutive,
            'valid_correspondence_threshold': 0.5,
            'trimming_ratio': 0.1,
            'min_distance_threshold': 5.0,
            'icp_error_threshold': icp_error_threshold,
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
        
        # Compute APE metrics (ape_basalt is already pre-computed)
        ape_corrected = compute_ape_metrics(gt_poses, poses_corrected[valid_index])
        
        # Save poses
        save_poses_to_file(poses_corrected, poses_path)
        save_ape_to_csv(ape_basalt, result_basalt_path)
        save_ape_to_csv(ape_corrected, result_corrected_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        with lock:
            print(f"Completed: seq={seq_str}, min_segment_size={min_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                  f"icp_error_threshold={icp_error_threshold}")
            print(f"  Basalt APE RMSE: {ape_basalt['rmse']:.4f}")
            print(f"  Corrected APE RMSE: {ape_corrected['rmse']:.4f}")
            print(f"  Execution time: {execution_time:.2f} seconds")
    
    except Exception as e:
        with lock:
            print(f"ERROR in seq={seq_str}, min_segment_size={min_segment_size}, "
                  f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                  f"icp_error_threshold={icp_error_threshold}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run basalt trajectory correction experiments')
    parser.add_argument('--basalt_pose_dir', type=str, required=True,
                        help='Directory containing basalt pose CSV files')
    parser.add_argument('--kitti_base_dir', type=str, required=True,
                        help='Base directory for KITTI dataset')
    parser.add_argument('--map_path', type=str, required=True,
                        help='Path to map points file (.npz)')
    parser.add_argument('--calib_path', type=str, required=True,
                        help='Path to calibration file for KITTI 360')
    parser.add_argument('--output_dir_results', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--n_threads', type=int, default=os.cpu_count()-2,
                        help='Number of processes to use (default: cpu_count-2)')
    
    args = parser.parse_args()
    
    # Parameter ranges
    min_segment_sizes = [50, 100, 150]
    knn_neighbors_list = [10, 20, 50, 100]
    max_error_consecutive_list = [10, 50, 100, 10000]
    icp_error_threshold_list = [1.0, 1.5, 2.0]
    
    # Sequences: 00-10, skipping 03
    sequences = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    
    # Create output directory
    os.makedirs(args.output_dir_results, exist_ok=True)
    
    # Compute tf_cam_to_velo once (same for all processes)
    tf_cam_to_velo = read_calib_file_kitti_360(args.calib_path)
    
    # Pre-compute sequence data for all sequences
    print("Pre-computing sequence data...")
    sequence_data = {}
    for seq in sequences:
        seq_str = f"{seq:02d}"
        print(f"  Pre-computing data for sequence {seq_str}...")
        try:
            sequence_data[seq] = precompute_sequence_data(
                seq, args.basalt_pose_dir, args.kitti_base_dir, args.map_path, tf_cam_to_velo
            )
        except Exception as e:
            print(f"  ERROR pre-computing data for sequence {seq_str}: {e}")
            continue
       
    print(f"Pre-computed data for {len(sequence_data)} sequences")
    
    # Generate all experiment combinations
    experiments = []
    for min_segment_size in min_segment_sizes:
        for knn_neighbors in knn_neighbors_list:
            for max_error_consecutive in max_error_consecutive_list:
                for icp_error_threshold in icp_error_threshold_list:
                    for seq in sequences:
                        experiments.append((
                            seq, min_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold
                        ))
    
    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")
    print(f"Using {args.n_threads} processes")
    print(f"Output directory: {args.output_dir_results}")
    
    # Process lock for synchronized printing (using Manager for picklable lock)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    
    # Run experiments with process pool
    with ProcessPoolExecutor(max_workers=args.n_threads) as executor:
        futures = []
        for seq, min_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold in experiments:
            # Skip if sequence data wasn't pre-computed successfully
            if seq not in sequence_data:
                continue
                
            # Pass a deep copy of sequence data to each process
            seq_data_copy = {
                'gt_poses': sequence_data[seq]['gt_poses'].copy(),
                'basalt_poses': sequence_data[seq]['basalt_poses'].copy(),
                'valid_index': sequence_data[seq]['valid_index'].copy(),
                'ape_basalt': copy.deepcopy(sequence_data[seq]['ape_basalt']),  # Dictionary needs deep copy
                'points_lane_map': sequence_data[seq]['points_lane_map']  # Map points can be shared
            }
            future = executor.submit(
                run_single_experiment,
                seq, min_segment_size, knn_neighbors, max_error_consecutive, icp_error_threshold,
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

