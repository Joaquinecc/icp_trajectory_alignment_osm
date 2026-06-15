# Copyright 2026 Distance Technologies Oy. For internal use only.
#
import argparse
import copy
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pykitti
from scipy.spatial.transform import Rotation
from utils_exp import compute_ape_metrics, save_ape_to_csv, save_poses_to_file

# Add workspace root to Python path so we can import osm_align from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # Go up from script/ to workspace root
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from osm_align.core.odometry_correction import OdomCorrector
from osm_align.utils import utils
from osm_align.utils.kitti_utils import angle_dict, cordinta_dict, read_kitti_pose


def precompute_basalt_sequence_data(
    seq: int, basalt_pose_dir: str, kitti_base_dir: str, map_path: str
) -> dict:
    """
    Pre-compute all sequence-specific data for basalt poses.

    Parameters
    ----------
    seq : int
        Sequence number
    basalt_pose_dir : str
        Directory containing basalt pose CSV files
    kitti_base_dir : str
        Base directory for KITTI dataset
    map_path : str
        Path to map points file

    Returns
    -------
    dict
        Dictionary containing pre-computed data
    """
    seq_str = f"{seq:02d}"

    # Load KITTI data
    kitti_odom = pykitti.odometry(kitti_base_dir, seq_str)

    # Load basalt poses
    basalt_pose_file_path = os.path.join(basalt_pose_dir, f"{seq_str}.csv")
    basalt_poses = utils.read_basalt_pose(basalt_pose_file_path)

    # Transform poses
    T_cam0_velo = kitti_odom.calib.T_cam0_velo
    tf_yaw_to_enu = np.eye(4)
    tf_yaw_to_enu[:3, :3] = Rotation.from_euler(
        "z", -angle_dict[seq_str], degrees=True
    ).as_matrix()

    # Transform GT poses
    gt_poses = np.array(kitti_odom.poses) @ T_cam0_velo
    gt_poses = tf_yaw_to_enu @ np.linalg.inv(gt_poses[0]) @ gt_poses

    # Transform basalt poses
    basalt_poses = basalt_poses @ T_cam0_velo
    basalt_poses = tf_yaw_to_enu @ np.linalg.inv(basalt_poses[0]) @ basalt_poses

    # Get map points
    new_origin_gps = [
        cordinta_dict[seq_str]["origin_lat"],
        cordinta_dict[seq_str]["origin_lon"],
    ]
    points_lane_map = utils.get_map_points(map_path, new_origin_gps)

    # Pre-compute APE metrics for basalt poses
    ape_baseline = compute_ape_metrics(gt_poses, basalt_poses)

    return {
        "gt_poses": gt_poses,
        "odom_poses": basalt_poses,
        "ape_baseline": ape_baseline,
        "points_lane_map": points_lane_map,
    }


def precompute_liodom_sequence_data(
    seq: int, liodom_pose_dir: str, kitti_base_dir: str
) -> dict:
    """
    Pre-compute all sequence-specific data for liodom poses.

    Parameters
    ----------
    seq : int
        Sequence number
    liodom_pose_dir : str
        Directory containing liodom pose files
    kitti_base_dir : str
        Base directory for KITTI dataset

    Returns
    -------
    dict
        Dictionary containing pre-computed data
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
    tf_yaw_to_enu[:3, :3] = Rotation.from_euler(
        "z", -angle_dict[seq_str], degrees=True
    ).as_matrix()

    # Transform GT poses
    gt_poses = np.array(kitti_odom.poses) @ T_cam0_velo
    gt_poses = tf_yaw_to_enu @ np.linalg.inv(gt_poses[0]) @ gt_poses

    # Transform liodom poses (already in velodyne frame, just apply yaw rotation)
    liodom_poses = tf_yaw_to_enu @ np.linalg.inv(liodom_poses[0]) @ liodom_poses

    # Get map points (sequence-specific map path)
    seq_map_path = os.path.join(
        kitti_base_dir, "map", seq_str, f"{seq_str}_map_points.npz"
    )
    new_origin_gps = [
        cordinta_dict[seq_str]["origin_lat"],
        cordinta_dict[seq_str]["origin_lon"],
    ]
    points_lane_map = utils.get_map_points(seq_map_path, new_origin_gps)

    # Verify poses have same length
    assert len(liodom_poses) == len(
        gt_poses
    ), f"Liodom poses ({len(liodom_poses)}) and GT poses ({len(gt_poses)}) must have same length"

    # Pre-compute APE metrics for liodom poses
    ape_baseline = compute_ape_metrics(gt_poses, liodom_poses)

    return {
        "gt_poses": gt_poses,
        "odom_poses": liodom_poses,
        "ape_baseline": ape_baseline,
        "points_lane_map": points_lane_map,
    }


def run_single_experiment(
    seq: int,
    min_segment_size: int,
    knn_neighbors: int,
    max_error_consecutive: int,
    icp_error_threshold: float,
    max_segment_size: int,
    experiment_folder: str,
    seq_data: dict,
    odom_type: str,
    lock: multiprocessing.Lock,
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
    max_segment_size : int
        Maximum segment size
    experiment_folder : str
        Path to the experiment folder (contains results/ and poses/ subfolders)
    seq_data : dict
        Pre-computed sequence data containing:
        - gt_poses: Ground truth poses
        - odom_poses: Odometry poses (basalt or liodom)
        - ape_baseline: Pre-computed APE metrics for baseline
        - points_lane_map: Map points
    odom_type : str
        Type of odometry ("basalt" or "liodom")
    lock : multiprocessing.Lock
        Process lock for synchronized printing
    """
    seq_str = f"{seq:02d}"

    try:
        # Start timing
        start_time = time.time()

        # Use pre-computed sequence data (passed as copy)
        gt_poses = seq_data["gt_poses"]
        odom_poses = seq_data["odom_poses"]
        ape_baseline = seq_data["ape_baseline"]
        points_lane_map = seq_data["points_lane_map"]

        # Define output paths
        results_folder = os.path.join(experiment_folder, "results")
        poses_folder = os.path.join(experiment_folder, "poses")

        poses_path = os.path.join(poses_folder, f"{seq_str}.txt")
        result_corrected_path = os.path.join(results_folder, f"{seq_str}.csv")

        # Check if experiment already completed for this sequence
        if os.path.exists(poses_path) and os.path.exists(result_corrected_path):
            with lock:
                print(
                    f"Skipping ({odom_type}, already completed): seq={seq_str}, "
                    f"min_segment_size={min_segment_size}, knn_neighbors={knn_neighbors}, "
                    f"max_error_consecutive={max_error_consecutive}, "
                    f"icp_error_threshold={icp_error_threshold}, max_segment_size={max_segment_size}"
                )
            return

        # Create subfolders if they don't exist
        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(poses_folder, exist_ok=True)

        # Setup correction parameters
        args = {
            "min_segment_size": min_segment_size,
            "knn_neighbors": knn_neighbors,
            "max_error_consecutive": max_error_consecutive,
            "valid_correspondence_threshold": 0.5,
            "trimming_ratio": 0.1,
            "min_distance_threshold": 10.0,
            "icp_error_threshold": icp_error_threshold,
            "max_segment_size": max_segment_size,
        }

        # Create corrector
        trajectory_correction = OdomCorrector(points_lane_map, args)

        # Apply correction
        poses_corrected = []
        for i in range(len(odom_poses)):
            pose_received = odom_poses[i]
            pose_corrected, message = trajectory_correction.apply(pose_received)
            poses_corrected.append(pose_corrected)

        poses_corrected = np.array(poses_corrected)

        # Compute APE metrics
        ape_corrected = compute_ape_metrics(gt_poses, poses_corrected)

        # Save poses and results
        save_poses_to_file(poses_corrected, poses_path)
        save_ape_to_csv(ape_corrected, result_corrected_path)

        # Calculate execution time
        execution_time = time.time() - start_time

        with lock:
            print(f"  Execution time: {execution_time:.2f} seconds")
            print(
                f"Completed ({odom_type}): seq={seq_str}, min_segment_size={min_segment_size}, "
                f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                f"icp_error_threshold={icp_error_threshold}, max_segment_size={max_segment_size}"
            )
            print(f"  Baseline APE RMSE: {ape_baseline['rmse']:.4f}")
            print(f"  Corrected APE RMSE: {ape_corrected['rmse']:.4f}")

    except Exception as e:
        with lock:
            print(
                f"ERROR ({odom_type}) in seq={seq_str}, min_segment_size={min_segment_size}, "
                f"knn_neighbors={knn_neighbors}, max_error_consecutive={max_error_consecutive}, "
                f"icp_error_threshold={icp_error_threshold}, max_segment_size={max_segment_size}: {e}"
            )


def setup_odom_directory(output_dir: str, odom_type: str) -> tuple[str, str]:
    """
    Create the directory structure for a given odometry type.

    Returns
    -------
    tuple[str, str]
        (experiments_dir, baseline_dir)
    """
    odom_dir = os.path.join(output_dir, odom_type)
    experiments_dir = os.path.join(odom_dir, "experiments")
    baseline_dir = os.path.join(odom_dir, "baseline")

    os.makedirs(odom_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    return experiments_dir, baseline_dir


def create_experiment_folders(experiments_dir: str, experiment_params: list) -> dict:
    """
    Create experiment folders for all parameter combinations.

    Returns
    -------
    dict
        Mapping from params tuple to experiment folder path
    """
    experiment_folders = {}
    for params in experiment_params:
        (
            min_segment_size,
            knn_neighbors,
            max_error_consecutive,
            icp_error_threshold,
            max_segment_size,
        ) = params
        folder_name = f"r_{min_segment_size}_{knn_neighbors}_{max_error_consecutive}_{icp_error_threshold}_{max_segment_size}"
        experiment_folder = os.path.join(experiments_dir, folder_name)
        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "poses"), exist_ok=True)
        experiment_folders[params] = experiment_folder

    return experiment_folders


def main():
    parser = argparse.ArgumentParser(
        description="Run trajectory correction experiments for basalt and liodom on KITTI"
    )
    parser.add_argument(
        "--basalt_pose_dir",
        type=str,
        required=True,
        help="Directory containing basalt pose CSV files",
    )
    parser.add_argument(
        "--liodom_pose_dir",
        type=str,
        required=True,
        help="Directory containing liodom pose files",
    )
    parser.add_argument(
        "--kitti_base_dir",
        type=str,
        required=True,
        help="Base directory for KITTI dataset",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        required=True,
        help="Path to map points file (.npz) for basalt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=os.cpu_count() - 2,
        help="Number of processes to use (default: cpu_count-2)",
    )

    args = parser.parse_args()

    # Parameter ranges
    min_segment_sizes = [50, 100, 150, 200, 300]
    knn_neighbors_list = [10, 20, 50, 100]
    max_error_consecutive_list = [10, 50, 100, 10000]
    icp_error_threshold_list = [1.0, 1.5, 2.0]
    max_segment_sizes = [1000, 500, 1500, 2000]

    # Sequences: 00-10, skipping 03
    sequences = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate all experiment parameter combinations
    experiment_params = []
    for min_segment_size in min_segment_sizes:
        for knn_neighbors in knn_neighbors_list:
            for max_error_consecutive in max_error_consecutive_list:
                for icp_error_threshold in icp_error_threshold_list:
                    for max_segment_size in max_segment_sizes:
                        experiment_params.append(
                            (
                                min_segment_size,
                                knn_neighbors,
                                max_error_consecutive,
                                icp_error_threshold,
                                max_segment_size,
                            )
                        )

    # Setup directory structures for both odometry types
    basalt_experiments_dir, basalt_baseline_dir = setup_odom_directory(
        args.output_dir, "basalt"
    )
    liodom_experiments_dir, liodom_baseline_dir = setup_odom_directory(
        args.output_dir, "liodom"
    )

    # Create experiment folders for both
    basalt_experiment_folders = create_experiment_folders(
        basalt_experiments_dir, experiment_params
    )
    liodom_experiment_folders = create_experiment_folders(
        liodom_experiments_dir, experiment_params
    )

    # Pre-compute sequence data for all sequences (both odometry types)
    print("Pre-computing sequence data...")

    basalt_sequence_data = {}
    liodom_sequence_data = {}

    for seq in sequences:
        seq_str = f"{seq:02d}"

        # Basalt
        print(f"  Pre-computing basalt data for sequence {seq_str}...")
        try:
            basalt_sequence_data[seq] = precompute_basalt_sequence_data(
                seq,
                args.basalt_pose_dir,
                args.kitti_base_dir,
                args.map_path,
            )
        except Exception as e:
            print(f"  ERROR pre-computing basalt data for sequence {seq_str}: {e}")

        # Liodom
        print(f"  Pre-computing liodom data for sequence {seq_str}...")
        try:
            liodom_sequence_data[seq] = precompute_liodom_sequence_data(
                seq,
                args.liodom_pose_dir,
                args.kitti_base_dir,
            )
        except Exception as e:
            print(f"  ERROR pre-computing liodom data for sequence {seq_str}: {e}")

    print(f"Pre-computed basalt data for {len(basalt_sequence_data)} sequences")
    print(f"Pre-computed liodom data for {len(liodom_sequence_data)} sequences")

    # Save baseline results
    print("Saving baseline results...")
    for seq, data in basalt_sequence_data.items():
        seq_str = f"{seq:02d}"
        baseline_path = os.path.join(basalt_baseline_dir, f"{seq_str}.csv")
        save_ape_to_csv(data["ape_baseline"], baseline_path)
        print(f"  Saved basalt baseline for sequence {seq_str}")

    for seq, data in liodom_sequence_data.items():
        seq_str = f"{seq:02d}"
        baseline_path = os.path.join(liodom_baseline_dir, f"{seq_str}.csv")
        save_ape_to_csv(data["ape_baseline"], baseline_path)
        print(f"  Saved liodom baseline for sequence {seq_str}")

    # Generate all experiment runs (params + sequence + odom_type combinations)
    experiments = []

    # Basalt experiments
    for params in experiment_params:
        for seq in sequences:
            if seq in basalt_sequence_data:
                experiments.append(("basalt", params, seq))

    # Liodom experiments
    for params in experiment_params:
        for seq in sequences:
            if seq in liodom_sequence_data:
                experiments.append(("liodom", params, seq))

    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")
    print(f"Using {args.n_threads} processes")
    print(f"Output directory: {args.output_dir}")

    # Process lock for synchronized printing (using Manager for picklable lock)
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # Run experiments with process pool
    with ProcessPoolExecutor(max_workers=args.n_threads) as executor:
        futures = []
        for odom_type, params, seq in experiments:
            (
                min_segment_size,
                knn_neighbors,
                max_error_consecutive,
                icp_error_threshold,
                max_segment_size,
            ) = params

            # Select the right data and folders based on odometry type
            if odom_type == "basalt":
                sequence_data = basalt_sequence_data
                experiment_folders = basalt_experiment_folders
            else:
                sequence_data = liodom_sequence_data
                experiment_folders = liodom_experiment_folders

            # Pass a deep copy of sequence data to each process
            seq_data_copy = {
                "gt_poses": sequence_data[seq]["gt_poses"].copy(),
                "odom_poses": sequence_data[seq]["odom_poses"].copy(),
                "ape_baseline": copy.deepcopy(sequence_data[seq]["ape_baseline"]),
                "points_lane_map": sequence_data[seq]["points_lane_map"],
            }

            future = executor.submit(
                run_single_experiment,
                seq,
                min_segment_size,
                knn_neighbors,
                max_error_consecutive,
                icp_error_threshold,
                max_segment_size,
                experiment_folders[params],
                seq_data_copy,
                odom_type,
                lock,
            )
            futures.append(future)

        # Wait for all experiments to complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            with lock:
                print(
                    f"Progress: {completed}/{total_experiments} experiments completed"
                )
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                with lock:
                    print(f"Experiment failed with exception: {e}")

    print("All experiments completed!")


if __name__ == "__main__":
    main()
