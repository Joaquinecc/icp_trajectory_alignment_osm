import os
import numpy as np
from typing import Dict, List, Tuple, Union
import math

angle_dict: Dict[str, float] = { #Initial Yaw orientation of the vehicle
    "00": -58.922619848964835,
    "01": 92.06246434076236,
    "02": -53.69354870803649,
    "04": -96.11460573874959,
    "05": -99.19843939674873,
    "06": 175.56880178138854,
    "07": 33.38621359499011,
    "08": -6.159108842836925,
    "09": 27.772955417117462,
    "10": 15.411929213671414
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

    seq_id_str = str(seq_id).zfill(2)
    if seq_id_str not in kitti_sequences:
        raise ValueError(f"Unknown KITTI sequence id: {seq_id}")
    date, drive, frames = kitti_sequences[seq_id_str]
    return date, drive, frames

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

