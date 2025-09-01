import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Union, Optional
# from geometry_msgs.msg import Pose

def pose_to_homogenous_matrix(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert a 2D rotation matrix and translation vector to a 4x4 homogeneous transformation matrix.
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix of shape (2, 2).
    T : np.ndarray
        Translation vector of shape (2,).
    Returns
    -------
    T_hom : np.ndarray
        Homogeneous transformation matrix of shape (4, 4).
    Examples
    --------
    >>> R = np.array([[0, 1], [-1, 0]])
    >>> T = np.array([1, 2])
    >>> T_hom = pose_to_homogenous_matrix(R, T)
    >>> print(T_hom)
    [[0 1 1 2]
     [-1 0 0 0]
     [0 0 1 0]
     [0 0 0 1]]
    """
    T_hom = np.eye(4)
    T_hom[:3, :3] = R
    T_hom[:3, 3] = T
    return T_hom


def find_interception_normal_shooting(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    map_points: np.ndarray
) -> np.ndarray:
    """
    Find closest intersections of trajectory normals with map segments using KNN neighbors.

    For each trajectory point, computes its tangent direction, shoots a normal
    (perpendicular) ray, and finds the closest intersection with line segments
    formed by K-nearest neighbor points in the map.

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array. Length should match trajectory_points.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points used to form line segments.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing the closest intersection points for each
        trajectory point. Points with no valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1], [0, -1], [1, -1]])
    >>> knn_idx = [[0, 1], [1, 2], [2, 1]]
    >>> intersections = find_interception_normal_shooting(trajectory, knn_idx, map_pts)
    >>> intersections.shape
    (3, 2)

    Notes
    -----
    The algorithm:
    1. Estimates tangent direction for each trajectory point using neighboring points
    2. Computes normal (perpendicular) direction
    3. Tests intersection with segments formed by consecutive map neighbors
    4. Returns the closest valid intersection within segment bounds [0, 1]
    """

    intercept_points = np.zeros_like(trajectory_points)

    for i, p in enumerate(trajectory_points):
        # Estimate tangent direction from trajectory
        if 0 < i < len(trajectory_points) - 1:
            tangent = trajectory_points[i + 1] - trajectory_points[i - 1]
        elif i < len(trajectory_points) - 1:
            tangent = trajectory_points[i + 1] - p
        elif i > 0:
            tangent = p - trajectory_points[i - 1]
        else:
            tangent = np.array([1.0, 0.0])
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular to tangent

        min_dist = np.inf
        best_intercept_point = np.nan

        knn_idx = knn_indices[i]
        for j in range(len(knn_idx)):
            map_idx = knn_idx[j]

            # Try previous neighbor
            if map_idx > 0:
                a = map_points[map_idx - 1]
                b = map_points[map_idx]
                ab = b - a
                A = np.column_stack((ab, -normal))
                det = np.linalg.det(A)
                if abs(det) > 1e-10:
                    # sol = np.linalg.solve(A, p - a)
                    sol= np.linalg.inv(A) @ (p-a)
                    # sol=  np.linalg.lstsq(A, p-a, rcond=None)[0]
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        dist = abs(s)
                        if dist < min_dist:
                            min_dist = dist
                            proj = a + t * ab
                            best_intercept_point = proj
            # Try next neighbor
            if map_idx + 1 < len(map_points):
                a = map_points[map_idx]
                b = map_points[map_idx + 1]
                ab = b - a
                A = np.column_stack((ab, -normal))
                det = np.linalg.det(A)
                if abs(det) > 1e-10:
                    # sol = np.linalg.solve(A, p - a)
                    sol= np.linalg.inv(A) @ (p-a)

                    # sol=  np.linalg.lstsq(A, p-a, rcond=None)[0]
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        dist = abs(s)
                        if dist < min_dist:
                            min_dist = dist
                            proj = a + t * ab
                            best_intercept_point = proj

        intercept_points[i] = best_intercept_point

    return intercept_points

def find_interception_normal_shooting_nextpoint_tangent(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    map_points: np.ndarray, 
    map_next_points: np.ndarray
) -> np.ndarray:
    """
    Find intersections of trajectory normals with map segments, prioritizing parallel tangents.

    For each trajectory point, shoots a normal ray and finds intersections with segments
    defined by map points and their next points. Selects the intersection where the
    segment's tangent direction is most parallel to the trajectory's tangent direction.

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points.
    map_next_points : np.ndarray
        Array of shape (M, 2) containing the next point for each map point,
        used to define line segments. Should have same length as map_points.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing intersection points. Points with no
        valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1]])
    >>> next_pts = np.array([[0.5, 1], [1.5, 1], [2.5, 1]])
    >>> knn_idx = [[0], [1], [2]]
    >>> result = find_interception_normal_shooting_nextpoint_tangent(
    ...     trajectory, knn_idx, map_pts, next_pts)
    >>> result.shape
    (3, 2)

    Notes
    -----
    This algorithm prioritizes intersections where the map segment direction
    is most parallel to the trajectory direction (parallel_score >= 0.7).
    If a very good match is found (parallel_score >= 0.9), the search stops early.
    """
    intercept_points = np.full_like(trajectory_points, np.nan)

    for i, p in enumerate(trajectory_points):
        # Estimate tangent direction from trajectory
        if 0 < i < len(trajectory_points) - 1:
            tangent_traj = trajectory_points[i + 1] - trajectory_points[i - 1]
        elif i < len(trajectory_points) - 1:
            tangent_traj = trajectory_points[i + 1] - p
        elif i > 0:
            tangent_traj = p - trajectory_points[i - 1]
        else:
            tangent_traj = np.array([1.0, 0.0])
        norm = np.linalg.norm(tangent_traj)
        if norm > 1e-10:
            tangent_traj = tangent_traj / norm
        else:
            tangent_traj = np.array([1.0, 0.0])
        normal_traj = np.array([-tangent_traj[1], tangent_traj[0]])  # Perpendicular to tangent

        best_parallel = -np.inf
        best_intercept_point = np.nan

        knn_idx = knn_indices[i]
        for j in range(len(knn_idx)):
            map_idx = knn_idx[j]
            a = map_points[map_idx]
            b = map_next_points[map_idx]
            ab = b - a
            # Check that ab is not degenerate
            ab_norm = np.linalg.norm(ab)
            if ab_norm < 1e-10:
                continue
            ab_unit = ab / ab_norm
            # Find intersection of normal_traj at p with the segment ab
            # Solve: a + t * ab = p + s * normal_traj
            # => t * ab - s * normal_traj = (p - a)
            A = np.column_stack((ab, -normal_traj))
            det = np.linalg.det(A)
            if abs(det) > 1e-10:
                sol = np.linalg.inv(A) @ (p - a)
                t, s = sol[0], sol[1]
                # Only accept intersection if t in [0,1] (segment)
                if 0.0 <= t <= 1.0:
                    # Compute parallelism (dot product, higher is more parallel)
                    parallel_score = abs(np.dot(ab_unit, tangent_traj))
                    if parallel_score > best_parallel and parallel_score>=0.7:
                        best_parallel = parallel_score
                        proj = a + t * ab
                        best_intercept_point = proj
                        if parallel_score>=0.95: #no need to check more, choose closest 
                            break

        intercept_points[i] = best_intercept_point

    return intercept_points

def solveIcp2d(
    source: np.ndarray, 
    target: np.ndarray, 
    max_iterations: int = 50, 
    tolerance: float = 1e-6
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
    N = src.shape[0]

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
    tolerance: float = 1e-8
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
        sorted_indices = np.argsort(distances)
        best_indices = sorted_indices[:N_trimmed]

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
        mean_error = np.mean(np.linalg.norm(src_points[best_indices] - tgt_points[best_indices], axis=1))

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate final icp_error using only the non-trimmed points (best correspondences)
    correct_source_points = (R_total @ source_points.T).T + t_total
    icp_error = np.mean(np.linalg.norm(correct_source_points[best_indices] - target_points[best_indices], axis=1))
    return R_total, t_total, icp_error

def solve_ransac_icp_2d(
    source_points: np.ndarray, 
    target_points: np.ndarray, 
    min_sample_size: int = 5, 
    max_iterations: int = 100, 
    distance_threshold: float = 2.0, 
    inlier_ratio_threshold: float = 0.1, 
    max_icp_iterations: int = 5, 
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Robust 2D point cloud registration using RANSAC with ICP refinement.

    Combines RANSAC sampling with ICP optimization to achieve robust registration
    in the presence of outliers. Uses weighted sampling favoring recent trajectory
    points and iteratively refines transformations using inlier sets.

    Parameters
    ----------
    source_points : np.ndarray
        Array of shape (N, 2) containing source points to be transformed.
    target_points : np.ndarray
        Array of shape (N, 2) containing target points to align to.
        Must have the same number of points as source.
    min_sample_size : int, default=5
        Minimum number of point pairs to sample for each RANSAC iteration.
        Must be at least 2 for 2D rigid transformation estimation.
    max_iterations : int, default=10000
        Maximum number of RANSAC iterations to attempt.
    distance_threshold : float, default=2.0
        Maximum Euclidean distance for a transformed source point to be
        considered an inlier with respect to its target point.
    inlier_ratio_threshold : float, default=0.4
        Minimum fraction of points that must be inliers for a transformation
        to be considered valid. Range: [0, 1].
    max_icp_iterations : int, default=5
        Maximum number of ICP iterations for each RANSAC candidate refinement.
    tolerance : float, default=1e-6
        ICP convergence tolerance for transformation refinement.

    Returns
    -------
    R_best : np.ndarray
        Array of shape (2, 2) containing the best rotation matrix found.
    t_best : np.ndarray
        Array of shape (2,) containing the best translation vector found.
    final_error : float
        Mean registration error computed using the final inlier set.

    Raises
    ------
    ValueError
        If the number of input points is less than min_sample_size.

    Examples
    --------
    >>> # Create data with outliers
    >>> source = np.random.randn(100, 2)
    >>> target = source + [1, 1] + 0.1 * np.random.randn(100, 2)
    >>> # Add some outliers
    >>> target[-10:] += 10 * np.random.randn(10, 2)
    >>> R, t, error = solve_ransac_icp_2d(source, target, min_sample_size=10)
    >>> print(f"Estimated translation: {t}")
    >>> print(f"Final error: {error:.4f}")

    Notes
    -----
    The algorithm:
    1. Randomly samples point subsets with higher probability for recent points
    2. Estimates transformation using mini-ICP on the sample
    3. Counts inliers using distance_threshold
    4. Refines transformation using all inliers if inlier_ratio is sufficient
    5. Falls back to trimmed ICP if no good RANSAC model is found
    
    The weighted sampling (favoring recent points) is particularly useful for
    trajectory alignment where recent observations are more reliable.
    """
    
    N = source_points.shape[0]
    if N < min_sample_size:
        raise ValueError(f"Need at least {min_sample_size} points, got {N}")
    
    best_inlier_count = 0
    best_R = np.eye(2)
    best_t = np.zeros(2)
    best_inliers = []
    best_error = np.inf
    weights = np.linspace(0.1, 1.0, N)
    weights=weights/weights.sum()
    for _ in range(max_iterations):
        # Randomly sample points
        # Sample indices with higher probability for indices closer to N (favoring recent points)
        sample_indices = np.random.choice(N, min_sample_size, replace=False, p=weights)
        sample_source = source_points[sample_indices]
        sample_target = target_points[sample_indices]
        
        # Compute transformation using sampled points with mini-ICP
        R_candidate, t_candidate, _ = solveIcp2d(
            sample_source, sample_target, 
            max_iterations=max_icp_iterations, 
            tolerance=tolerance
        )
        
        # Transform all source points using candidate transformation
        transformed_source = (R_candidate @ source_points.T).T + t_candidate
        
        # Compute distances to target points
        distances = np.linalg.norm(transformed_source - target_points, axis=1)
        
        # Find inliers
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)
        inlier_ratio = inlier_count / N
        
        # Check if this is a good candidate
        if (inlier_ratio >= inlier_ratio_threshold and 
            inlier_count > best_inlier_count):
            
            # Refine transformation using all inliers
            inlier_indices = np.where(inlier_mask)[0]
            inlier_source = source_points[inlier_indices]
            inlier_target = target_points[inlier_indices]
        
            R_refined, t_refined, refined_error = solveIcp2d(
                inlier_source, inlier_target,
                max_iterations=max_icp_iterations,
                tolerance=tolerance
            )
            
            # Update best solution
            best_inlier_count = inlier_count
            best_R = R_refined
            best_t = t_refined
            best_inliers = inlier_indices
            best_error = refined_error
            
        
          
    # # If no good model found, fall back to standard ICP
    # if best_inlier_count == 0:
    #     # print("Warning: RANSAC failed to find good model, falling back to standard ICP")
    #     best_R, best_t, final_error =  solve_trimmed_icp_2d(source_points, target_points, max_iterations=max_icp_iterations, tolerance=tolerance)
    
    # # # Calculate final error using best inliers
    final_transformed_source = (best_R @ source_points.T).T + best_t
    final_error = np.mean(np.linalg.norm(final_transformed_source[best_inliers] - target_points[best_inliers], axis=1))
    
    return best_R, best_t, final_error

def find_k_closest_neighbors_for_points(
    map_points_org: Union[List, np.ndarray], 
    points: np.ndarray, 
    k: int = 3
) -> np.ndarray:
    """
    Find K nearest neighbor indices for each query point in a map point cloud.

    Uses Euclidean distance in 2D (x, y coordinates) to find the closest map points
    for each query point, returning the indices for further processing.

    Parameters
    ----------
    map_points_org : list or np.ndarray
        Map points as array-like of shape (M, 2) or (M, 3). Only the first two
        dimensions (x, y) are used for distance computation.
    points : np.ndarray
        Query points as array of shape (N, 2) or (N, 3) in the same coordinate
        system as map_points. Only x, y coordinates are used.
    k : int, default=3
        Number of closest neighbors to find for each query point.

    Returns
    -------
    closest_indices : np.ndarray
        Array of shape (N, k) containing the indices of the k closest map points
        for each query point. Indices refer to positions in map_points_org.

    Examples
    --------
    >>> map_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]])
    >>> query_points = np.array([[0.5, 0.5], [1.5, 1.5]])
    >>> knn_indices = find_k_closest_neighbors_for_points(map_points, query_points, k=2)
    >>> print(f"Shape: {knn_indices.shape}")
    >>> print(f"Closest to [0.5, 0.5]: indices {knn_indices[0]}")
    Shape: (2, 2)
    Closest to [0.5, 0.5]: indices [0 3]

    Notes
    -----
    This function uses scipy's cdist for efficient distance computation.
    The returned indices can be used with other functions in this module
    that require KNN information for trajectory-to-map alignment.
    
    For 3D input points, only the first two dimensions are considered,
    making this suitable for 2D mapping applications where z-coordinates
    may vary due to terrain but x-y alignment is the primary concern.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    map_points = np.array(map_points_org)
    points = np.array(points)
    closest_indices = []

    for i in range(len(points)):
        query_point = np.array(points[i][:2])  # Use only x, y
        map_points_xy = map_points[:, :2]      # Only x, y for distance
        dists = cdist([query_point], map_points_xy)[0]
        k_indices = np.argsort(dists)[:k]
        # Return the indices
        closest_indices.append(k_indices)
    return np.array(closest_indices)
