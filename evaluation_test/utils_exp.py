from evo.main_ape import ape
from evo.core.trajectory import PosePath3D, Plane
from evo.core import metrics
import numpy as np
import csv
from typing import Dict
import copy

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
