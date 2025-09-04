import os
import numpy as np
from typing import Dict, List, Tuple, Union
import math

# Type hints for the configuration dictionaries
# angle_dict: Dict[str, float] = { #This was computed by the function rotation_alignment_utm_to_kitti_gt
#     "00": -58.922619848964835,
#     "01": 92.06246434076236,
#     "02": -53.69354870803649,
#     "04": -96.11460573874959,
#     "05": -99.19843939674873,
#     "06": 175.56880178138854,
#     "07": 33.38621359499011,
#     "08": -6.159108842836925,
#     "09": 27.772955417117462,
#     "10": 15.411929213671414
# }
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
    "10": 16.0}
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




def rotation_angle_2d(ref_point, target_point):
    """
    Compute the 2D rotation angle (degrees) between two vectors.
    
    Parameters
    ----------
    ref_point : tuple or list (x, y)
        The reference vector in the first frame.
    target_point : tuple or list (x, y)
        The corresponding vector in the rotated frame.
    
    Returns
    -------
    float
        Rotation angle in degrees (positive = counterclockwise).
    """
    x, y = ref_point
    xp, yp = target_point

    # dot and cross (scalar in 2D)
    dot = x * xp + y * yp
    cross = x * yp - y * xp

    # norms
    norm_ref = math.hypot(x, y)
    norm_target = math.hypot(xp, yp)

    # cosine and sine of the angle
    cos_theta = dot / (norm_ref * norm_target)
    sin_theta = cross / (norm_ref * norm_target)

    # clamp cos_theta for numerical safety
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    angle_rad = math.atan2(sin_theta, cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg



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


def rotation_alignment_utm_to_kitti_gt(kitti_raw_dir, kitti_gt_dir):
    """
    Compute the mean 2D rotation angle between UTM-projected GPS points and KITTI ground truth poses for each sequence.

    This function aligns UTM-projected GPS coordinates with the corresponding KITTI ground truth (GT) poses
    for a set of standard KITTI odometry sequences. For each sequence, it computes the mean 2D rotation angle
    (in degrees) between the UTM-projected GPS points and the translation component of the GT pose, after
    applying the standard KITTI coordinate transformation.

    Parameters
    ----------
    kitti_raw_dir : str
        Path to the root directory containing raw KITTI data (as required by pykitti).
    kitti_gt_dir : str
        Path to the directory containing KITTI ground truth pose files (one file per sequence, e.g., '00.txt').

    Returns
    -------
    frame_angle : dict
        Dictionary mapping KITTI sequence IDs (str) to the mean 2D rotation angle (float, in degrees)
        between UTM-projected GPS points and KITTI GT poses for that sequence. If no angles are computed
        for a sequence, the value will be NaN.

    Notes
    -----
    - The function uses the following KITTI sequences: '00', '01', '02', '04', '05', '06', '07', '08', '09', '10'.
    - The UTM projection is initialized using the first GPS coordinate of each sequence as the origin.
    - The KITTI GT pose is transformed using the standard KITTI rotation matrix before comparison.
    - The 2D rotation angle is computed using the `rotation_angle_2d` function, comparing the (x, y) components
      of the UTM-projected GPS point and the GT pose translation.
    - Requires the following packages: pykitti, lanelet2, numpy.

    Examples
    --------
    >>> angles = rotation_alignment_utm_to_kitti_gt("/path/to/kitti/raw", "/path/to/kitti/gt")
    >>> for seq, angle in angles.items():
    ...     print(f"Sequence {seq}: mean angle = {angle:.2f} degrees")
    Sequence 00: mean angle = -0.12 degrees
    Sequence 01: mean angle = 0.05 degrees
    ...

    """
    import pykitti
    import lanelet2
    from lanelet2.core import GPSPoint

    BASE_GT_PATH = kitti_gt_dir
    kitti_dir = kitti_raw_dir

    # KITTI coordinate system rotation matrix (from camera to world)
    R_kitti = np.array([
        [0,  0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    T_kitti = np.eye(4)
    T_kitti[:3, :3] = R_kitti

    # List of KITTI odometry sequence IDs to process
    frames_id = ['00', '01', '02', '04', '05', '06', '07', '08', '09', '10']
    frame_angle = {}

    for frame_id in frames_id:
        # Retrieve KITTI sequence metadata
        date, drive, frames = get_kitti_sequence_info(frame_id)
        frame_range = list(range(frames[0], frames[1] + 1))
        # Load raw KITTI data for the sequence
        kitti_raw = pykitti.raw(kitti_dir, date, drive, frames=frame_range)
        # Extract GPS coordinates (latitude, longitude, altitude)
        gps_coords = np.array([[x.packet.lat, x.packet.lon, x.packet.alt] for x in kitti_raw.oxts])
        # Initialize UTM projector using the first GPS coordinate as the origin
        origin = gps_coords[0]
        proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(origin[0], origin[1], origin[2]))
        # Load ground truth poses for the sequence
        gt_path_pose = os.path.join(BASE_GT_PATH, f'{frame_id}.txt')
        gt_poses = get_pose(gt_path_pose)   
        # Transform GT poses to KITTI world coordinates
        gt_poses = np.matmul(T_kitti, gt_poses)  # shape (N, 4, 4)
        angles = []
        # Compute 2D rotation angle for each frame (excluding the first)
        for i in range(1, len(gt_poses)):
            gps_pt = gps_coords[i]
            point = proj.forward(GPSPoint(gps_pt[0], gps_pt[1], gps_pt[2]))
            ref_point = [point.x, point.y, point.z]
            target_point = gt_poses[i][:3, -1]
            angle = rotation_angle_2d(ref_point[:2], target_point[:2])
            angles.append(angle)
        # Compute mean angle for the sequence
        if angles:
            mean_angle = np.mean(angles)
        else:
            mean_angle = float('nan')
        frame_angle[frame_id] = mean_angle

    return frame_angle
