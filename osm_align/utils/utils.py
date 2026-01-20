from typing import List, Optional, Tuple

import lanelet2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R


def tf_matrix_from(buffer, target: str, source: str, timeout_sec: float = 1.0):
    """
    Look up TF from source → target and return 4x4 homogeneous transform matrix.
    Equivalent to buffer.lookup_transform(target, source, ...).
    """
    import rclpy
    from geometry_msgs.msg import TransformStamped

    try:
        tf: TransformStamped = buffer.lookup_transform(
            target,
            source,
            rclpy.time.Time(seconds=0),  # latest available
            timeout=rclpy.duration.Duration(seconds=timeout_sec),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get transform {source}->{target}: {e}")

    # translation
    t = np.array(
        [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
    )

    # rotation
    q = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
    )

    # use scipy Rotation to get rotation matrix
    R_mat = R.from_quat(q).as_matrix()

    # build homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def pose_to_4x4(pose) -> np.ndarray:
    """
    Convert geometry_msgs/Pose to a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    pose : geometry_msgs.msg.Pose
        Input pose with position and orientation (quaternion).

    Returns
    -------
    numpy.ndarray
        A 4x4 homogeneous matrix in row-major layout.
    """
    quat = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=float,
    )

    M = np.eye(4)
    M[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    M[:3, :3] = Rotation.from_quat(quat).as_matrix()
    return M


def solveIcp2d(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve 2D Iterative Closest Point (ICP) registration between point sets.

    Estimates the optimal rigid transformation (rotation and translation) that
    aligns source points to target points using the standard ICP algorithm
    with SVD-based closed-form solution.

    Parameters
    ----------
    source : np.ndarray
        Array of shape (N, 2) containing source points to be transformed.
    target : np.ndarray
        Array of shape (N, 2) containing target points to align to.
        Must have the same number of points as source.
    max_iterations : int, default=50
        Maximum number of ICP iterations before termination.
    tolerance : float, default=1e-6
        Convergence tolerance. Algorithm stops when the change in mean error
        between iterations is less than this value.

    Returns
    -------
    R_total : np.ndarray
        Array of shape (2, 2) containing the optimal rotation matrix.
    T_total : np.ndarray
        Array of shape (2,) containing the optimal translation vector.
    final_error : float
        Mean Euclidean distance between transformed source and target points.

    Examples
    --------
    >>> source = np.array([[0, 0], [1, 0], [0, 1]])
    >>> target = np.array([[1, 1], [2, 1], [1, 2]])  # Translated by [1, 1]
    >>> R, T, error = solveIcp2d(source, target)
    >>> print(f"Translation: {T}, Error: {error:.6f}")
    Translation: [1. 1.], Error: 0.000000

    Notes
    -----
    This implementation uses SVD-based pose estimation in each iteration and
    accumulates the total transformation. The algorithm assumes one-to-one
    correspondence between source and target points.
    """
    src = np.copy(source)
    tgt = np.copy(target)

    R_total = np.eye(2)
    T_total = np.zeros((2,))

    prev_error = np.inf

    for i in range(max_iterations):
        # Compute centroids
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt, axis=0)

        # Center the points
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        # Compute covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        T = centroid_tgt - R @ centroid_src

        # Apply transformation
        src = (R @ src.T).T + T

        # Accumulate transformation
        R_total = R @ R_total
        T_total = R @ T_total + T

        # Compute mean error
        error = np.linalg.norm(src - tgt, axis=1)
        mean_error = np.mean(error)

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Final error: mean registration error (scalar)
    transformed_source = (R_total @ source.T).T + T_total
    final_error = np.mean(np.linalg.norm(transformed_source - target, axis=1))

    return R_total, T_total, final_error


def solve_trimmed_icp_2d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    trimming_ratio: float = 0.1,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Robust 2D ICP registration using trimmed least squares to handle outliers.

    Performs ICP registration while automatically removing the worst-fitting
    point correspondences (outliers) based on a specified trimming ratio.
    This makes the algorithm more robust to noise and incorrect correspondences.

    Parameters
    ----------
    source_points : np.ndarray
        Array of shape (N, 2) containing source points to be transformed.
    target_points : np.ndarray
        Array of shape (N, 2) containing target points to align to.
        Must have the same number of points as source.
    trimming_ratio : float, default=0.1
        Fraction of correspondences to trim (remove as outliers), in range [0, 1).
        For example, 0.1 means remove the worst 10% of correspondences.
    max_iterations : int, default=50
        Maximum number of ICP iterations before termination.
    tolerance : float, default=1e-8
        Convergence tolerance on mean error change between iterations.

    Returns
    -------
    R_total : np.ndarray
        Array of shape (2, 2) containing the optimal rotation matrix.
    t_total : np.ndarray
        Array of shape (2,) containing the optimal translation vector.
    icp_error : float
        Mean Euclidean distance between transformed source and target points
        using only the best (non-trimmed) correspondences.

    Examples
    --------
    >>> # Create data with outliers
    >>> source = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])  # Last point is outlier
    >>> target = np.array([[1, 1], [2, 1], [1, 2], [15, 5]])   # Corresponding outlier
    >>> R, t, error = solve_trimmed_icp_2d(source, target, trimming_ratio=0.25)
    >>> print(f"Translation: {t}, Error: {error:.6f}")
    Translation: [1. 1.], Error: 0.000000

    Notes
    -----
    The algorithm iteratively:
    1. Sorts correspondences by distance
    2. Keeps only the best (1 - trimming_ratio) fraction
    3. Computes transformation using trimmed correspondences
    4. Applies transformation and repeats

    This is particularly useful for outdoor robotics applications where
    sensor noise and dynamic objects can create spurious correspondences.
    """

    src_points = np.copy(source_points)
    tgt_points = np.copy(target_points)

    prev_error = np.inf
    N = src_points.shape[0]
    N_trimmed = int(N * (1.0 - trimming_ratio))

    R_total = np.eye(2)
    t_total = np.zeros(2)
    mean_error = 0.0
    best_indices = []

    for _ in range(max_iterations):
        # Compute distances and sort to find best correspondences
        distances = np.linalg.norm(src_points - tgt_points, axis=1)
        # sorted_indices = np.argsort(distances)
        # best_indices = sorted_indices[:N_trimmed]

        best_indices = np.argpartition(distances, N_trimmed)[:N_trimmed]

        src_trimmed = src_points[best_indices]
        tgt_trimmed = tgt_points[best_indices]

        # Compute centroids
        centroid_src = np.mean(src_trimmed, axis=0)
        centroid_tgt = np.mean(tgt_trimmed, axis=0)

        # Center the points
        src_centered = src_trimmed - centroid_src
        tgt_centered = tgt_trimmed - centroid_tgt

        # Compute cross-covariance
        W = src_centered.T @ tgt_centered

        # SVD for optimal rotation
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        # Ensure proper rotation (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_tgt - R @ centroid_src

        # Update cumulative transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Apply transformation to all src_points for next iteration
        src_points = (R @ src_points.T).T + t

        # Compute mean error using trimmed points
        mean_error = np.mean(
            np.linalg.norm(src_points[best_indices] - tgt_points[best_indices], axis=1)
        )

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate final icp_error using only the non-trimmed points (best correspondences)
    correct_source_points = (R_total @ source_points.T).T + t_total
    icp_error = np.mean(
        np.linalg.norm(
            correct_source_points[best_indices] - target_points[best_indices], axis=1
        )
    )
    return R_total, t_total, icp_error


def kabsch_2d(
    source_points: np.ndarray, target_points: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute optimal 2D rotation matrix using Kabsch algorithm.

    Calculates the optimal rotation matrix that aligns source points to target points
    in 2D space using singular value decomposition (SVD). The algorithm finds the
    rotation that minimizes the sum of squared distances between corresponding points.

    Parameters
    ----------
    source_points : np.ndarray
        Array of shape (N, 2) containing source points to be rotated.
    target_points : np.ndarray
        Array of shape (N, 2) containing target points to align to.
        Must have the same number of points as source_points.

    Returns
    -------
    np.ndarray or None
        2x2 rotation matrix that transforms source_points to align with target_points,
        or None if calculation fails (e.g., insufficient points).

    Examples
    --------
    >>> source = np.array([[0, 0], [1, 0], [0, 1]])
    >>> target = np.array([[0, 0], [0, 1], [-1, 0]])  # Rotated 90 degrees
    >>> R = kabsch_2d(source, target)
    >>> print(R)
    [[ 0.  1.]
     [-1.  0.]]

    Notes
    -----
    The Kabsch algorithm computes the optimal rotation by:
    1. Centering both point sets
    2. Computing the cross-covariance matrix
    3. Performing SVD to extract the rotation
    4. Ensuring a proper rotation (determinant = 1)

    This is commonly used for point cloud registration and trajectory alignment.
    """
    if source_points.shape[0] < 2 or target_points.shape[0] < 2:
        return None

    # Center both point sets
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute covariance matrix H = source_centered^T @ target_centered
    H = source_centered.T @ target_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix R = Vt^T @ U^T
    R = Vt.T @ U.T

    # Ensure proper rotation (determinant = 1)
    # If det < 0, we need to flip one column
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    return R


def lanelet_points_and_neighbour(
    lanelet_map: lanelet2.core.LaneletMap, min_dist: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the lanelet point list and its next-point associations.
    It ignores points that are too close to the previous point, defined by min_dist.

    Parameters
    ----------
    lanelet_map : lanelet2.LaneletMap
        Lanelet map to build the kdtree from.
    min_dist : float, default=3.0
        Minimum distance between points to be considered as a new point.

    Returns
    -------
    lane_points : np.ndarray
        Array of shape (N, 2) containing the lane points.
    lane_points_neighbour : np.ndarray
        Array of shape (N, 2) containing the next lane points.
    """
    d2 = (
        float(min_dist) ** 2
    )  # Is cheeaper to compare squared distances than the actual distances
    lane_points_neighbour = []
    lane_points = []
    lanelet_direction_points = []

    for lanelet in lanelet_map.laneletLayer:

        centerline_points = np.array(
            [(p.x, p.y) for p in lanelet.centerline], dtype=np.float64
        )
        # Distance-based thinning in one pass (compare squared distances)
        keep_idx = [0]
        last = centerline_points[0]
        for i in range(1, centerline_points.shape[0]):
            v = centerline_points[i]
            dx = v[0] - last[0]
            dy = v[1] - last[1]
            if dx * dx + dy * dy >= d2:  # Square distance comparison
                keep_idx.append(i)
                last = v

        centerline_points = centerline_points[keep_idx]

        # Create next-point associations for tangent computation
        neighbours = centerline_points.copy()
        neighbours[:-1] = centerline_points[1:]  # forward neighbour
        if len(centerline_points) > 1:
            neighbours[-1] = centerline_points[-2]  # last points to previous
        diffs = (
            neighbours - centerline_points
        )  # Always diff with ther consecutive point
        diffs[-1] = (
            centerline_points[-1] - neighbours[-1]
        )  # Except the last point, which is diff with the previous point
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)  # avoid division by zero
        tangents = diffs / norms

        lane_points.extend(centerline_points)
        lanelet_direction_points.extend(tangents)
        lane_points_neighbour.extend(neighbours)

    return np.hstack((lane_points, lane_points_neighbour, lanelet_direction_points))


def read_basalt_pose(file_path: str) -> List[np.ndarray]:
    """
    Read poses from a Basalt CSV file and return a list of 4x4 transformation matrices.

    The CSV file format is:
    #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
    with quaternion in (w, x, y, z) format.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing pose data.

    Returns
    -------
    List[np.ndarray]
        List of 4x4 homogeneous transformation matrices, one for each pose in the file.

    Examples
    --------
    >>> poses = read_basalt_pose("seq00.csv")
    >>> print(f"Number of poses: {len(poses)}")
    >>> print(f"First pose shape: {poses[0].shape}")
    Number of poses: 13
    First pose shape: (4, 4)
    """
    # Read CSV file, skipping header line (starts with #)
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=np.float64)

    poses = []
    for row in data:
        # Extract position: p_RS_R_x, p_RS_R_y, p_RS_R_z
        position = np.array([row[1], row[2], row[3]])

        # Extract quaternion: q_RS_w, q_RS_x, q_RS_y, q_RS_z
        # scipy expects (x, y, z, w) format
        quat_wxyz = np.array([row[5], row[6], row[7], row[4]])

        # Convert quaternion to rotation matrix using scipy
        rotation_matrix = Rotation.from_quat(quat_wxyz).as_matrix()

        # Build 4x4 homogeneous transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position

        poses.append(pose_matrix)

    return np.array(poses)


def get_map_points(map_path, new_origin_gps):
    """
    It loads the map points and updates them to the new origin.
    The map data it include the lane points information as the origin GPS coordinates.
    We update the lane points to the new origin.

    Parameters
    ----------
    map_path : str
        Path to the map file.
    new_origin_gps : tuple
        New origin GPS coordinates.

    Returns
    -------
    points_lane_map : np.ndarray
        Array of shape (N, 6) containing the lane points and their next points.

    Examples
    --------
    >>> points_lane_map = get_map_points("map.npz", (48.98254523586602, 8.39036610004500))
    >>> print(points_lane_map.shape)
    (N, 6)
    >>> print(points_lane_map[0])
    [x1, y1, x2, y2, x3, y3]
    """
    # Initialize odometry corrector
    lat0, lon0 = new_origin_gps
    loaded = np.load(map_path)
    points_lane_map = loaded["points_lane_map"]
    gps_origin_map = loaded["origin_gps"]
    map_projector = lanelet2.projection.UtmProjector(
        lanelet2.io.Origin(gps_origin_map[0], gps_origin_map[1])
    )

    # Calculate offset to new origin
    offset_xy = map_projector.forward(lanelet2.core.GPSPoint(lat0, lon0))

    offset_xy = np.array([offset_xy.x, offset_xy.y])
    # Update lane points, to new origin.
    points_lane_map[:, :2] = points_lane_map[:, :2] - offset_xy
    points_lane_map[:, 2:4] = points_lane_map[:, 2:4] - offset_xy

    return points_lane_map
