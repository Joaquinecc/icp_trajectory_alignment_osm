import os
import numpy as np
from typing import Dict, List, Tuple, Union

# Type hints for the configuration dictionaries
angle_dict: Dict[str, float] = {
    "00": -59.0,
    "01": 92.3,
    "02": -53.5,
    "04": -96.0,
    "05": -99.0,
    "06": 175.5,
    "07": 33.0,
    "08": -6.0,
    "09": 28.0,
    "10": 16.0
}
 
# Dictionary mapping KITTI frame IDs to origin lat/lon (angle_corr omitted)
cordinta_dict: Dict[str, Dict[str, float]] = {
    "00": {
        "origin_lat": 48.98254523586602,
        "origin_lon": 8.39036610004500,
    },
    "01": {
        "origin_lat": 49.006719195871,
        "origin_lon": 8.4893558806503,
    },
    "02": {
        "origin_lat": 48.987607723096,
        "origin_lon": 8.4697469732634,
    },
    "04": {
        "origin_lat": 49.033603440345,
        "origin_lon": 8.3950031909457,
    },
    "05": {
        "origin_lat": 49.04951961077,
        "origin_lon": 8.3965961639946,
    },
    "06": {
        "origin_lat": 49.05349304789598,
        "origin_lon": 8.39721998765449,
    },
    "07": {
        "origin_lat": 48.98523696217,
        "origin_lon": 8.3936414564418,
    },
    "08": {
        "origin_lat": 48.984262765672,
        "origin_lon": 8.3976660698392,
    },
    "09": {
        "origin_lat": 48.972104544468,
        "origin_lon": 8.4761469953335,
    },
    "10": {
        "origin_lat": 48.97253396005,
        "origin_lon": 8.4785980847297,
    },
}


def get_kitti_sequence_info(seq_id: Union[int, str]) -> Tuple[str, str, List[int]]:
    """
    Retrieve KITTI sequence metadata including date, drive number, and frame range.

    Maps KITTI sequence identifiers to their corresponding dataset metadata,
    providing the information needed to load raw KITTI data using pykitti.

    Parameters
    ----------
    seq_id : int or str
        KITTI sequence identifier (e.g., 0, "00", 1, "01"). Will be zero-padded
        to 2 digits for lookup.

    Returns
    -------
    date : str
        Date string in YYYY_MM_DD format (e.g., "2011_10_03").
    drive : str
        Drive number as zero-padded 4-digit string (e.g., "0027").
    frames : list of int
        Two-element list [start_frame, end_frame] indicating the valid frame range.

    Raises
    ------
    ValueError
        If the sequence ID is not found in the KITTI sequence database.

    Examples
    --------
    >>> date, drive, frames = get_kitti_sequence_info("00")
    >>> print(f"Date: {date}, Drive: {drive}, Frames: {frames}")
    Date: 2011_10_03, Drive: 0027, Frames: [0, 4540]
    
    >>> date, drive, frames = get_kitti_sequence_info(5)
    >>> print(f"Sequence 05: {date}/{drive}, {frames[1]-frames[0]+1} frames")
    Sequence 05: 2011_09_30/0018, 2761 frames
    """
    # Table of sequences
    kitti_sequences = {
        "00": ("2011_10_03", "0027", [0, 4540]),
        "01": ("2011_10_03", "0042", [0, 1100]),
        "02": ("2011_10_03", "0034", [0, 4660]),
        "03": ("2011_09_26", "0067", [0, 800]),
        "04": ("2011_09_30", "0016", [0, 270]),
        "05": ("2011_09_30", "0018", [0, 2760]),
        "06": ("2011_09_30", "0020", [0, 1100]),
        "07": ("2011_09_30", "0027", [0, 1100]),
        "08": ("2011_09_30", "0028", [1100, 5170]),
        "09": ("2011_09_30", "0033", [0, 1590]),
        "10": ("2011_09_30", "0034", [0, 1200]),
    }
    seq_id_str = str(seq_id).zfill(2)
    if seq_id_str not in kitti_sequences:
        raise ValueError(f"Unknown KITTI sequence id: {seq_id}")
    date, drive, frames = kitti_sequences[seq_id_str]
    return date, drive, frames

def odom_pose(odom_dir: str) -> np.ndarray:
    """
    Load KITTI odometry ground truth poses from a text file.

    Reads pose data in KITTI odometry format where each line contains 12 values
    representing a 3x4 transformation matrix (rotation and translation), and 
    converts them to 4x4 homogeneous transformation matrices.

    Parameters
    ----------
    odom_dir : str
        Path to the odometry file containing ground truth poses. Each line should
        contain 12 space-separated floats representing a 3x4 transformation matrix
        in row-major order.

    Returns
    -------
    transformation_matrices : np.ndarray
        Array of shape (N, 4, 4) containing homogeneous transformation matrices,
        where N is the number of poses in the file. Each matrix represents the
        pose of the vehicle at that frame.

    Raises
    ------
    FileNotFoundError
        If the specified odometry file does not exist. Ground truth odometry
        is only available for 10 sequences in KITTI.

    Examples
    --------
    >>> poses = odom_pose("/path/to/kitti/poses/00.txt")
    >>> print(f"Loaded {len(poses)} poses")
    >>> print(f"First pose translation: {poses[0][:3, 3]}")
    Loaded 4541 poses
    First pose translation: [0. 0. 0.]
    
    Notes
    -----
    The KITTI odometry format stores poses as 3x4 matrices in the form:
    [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]
    This function converts them to standard 4x4 homogeneous matrices by
    adding the bottom row [0 0 0 1].
    """
    if not os.path.exists(odom_dir):
        raise FileNotFoundError(f"Odom directory not found: {odom_dir}, Ground truth(Odometry) is available for only 10 sequences in KITTI. Stopping the process.")
    with open(odom_dir, 'r') as file:
        lines = file.readlines()
    transformation_data = [[float(val) for val in line.split()] for line in lines]
    homogenous_matrix_arr = []
    for i in range(len(transformation_data)):
        homogenous_matrix = np.identity(4)
        homogenous_matrix[0, :] = transformation_data[i][0:4]
        homogenous_matrix[1:2, :] = transformation_data[i][4:8]
        homogenous_matrix[2:3, :] = transformation_data[i][8:12]
        homogenous_matrix_arr.append(homogenous_matrix)
    return np.array(homogenous_matrix_arr)

def get_pose(path: str) -> np.ndarray:
    """
    Load pose data from a text file in 3x4 matrix format.

    Reads pose data where each line contains 12 space-separated values representing
    a 3x4 transformation matrix, and converts them to 4x4 homogeneous matrices
    by appending the bottom row [0, 0, 0, 1].

    Parameters
    ----------
    path : str
        Path to the pose file. Each line should contain 12 space-separated floats
        representing a 3x4 transformation matrix in row-major order.

    Returns
    -------
    poses : np.ndarray
        Array of shape (N, 4, 4) containing homogeneous transformation matrices,
        where N is the number of poses in the file.

    Examples
    --------
    >>> poses = get_pose("/path/to/poses.txt")
    >>> print(f"Loaded {len(poses)} poses")
    >>> print(f"First pose shape: {poses[0].shape}")
    Loaded 1000 poses
    First pose shape: (4, 4)
    
    >>> # Extract translation from first pose
    >>> translation = poses[0][:3, 3]
    >>> print(f"First pose translation: {translation}")
    First pose translation: [1.5 2.3 0.1]

    Notes
    -----
    This function is similar to `odom_pose` but uses a different internal
    implementation for reading the pose matrices. Both functions expect
    the same input format (12 values per line) and produce the same output
    format (4x4 homogeneous matrices).
    """
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        aux = [float(x) for x in line.split()]
        aux = np.array(aux).reshape(3, 4)
        aux = np.vstack([aux, [0, 0, 0, 1]])
        poses.append(aux)
    return np.array(poses)