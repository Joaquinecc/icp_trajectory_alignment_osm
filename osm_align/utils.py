import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Union, Optional
# from geometry_msgs.msg import Pose


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
                        if parallel_score>=0.9: #no need to check more, choose closest 
                            break

        intercept_points[i] = best_intercept_point

    return intercept_points

def find_interception_normal_shooting_parallel_tangent(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    tangents_map: np.ndarray, 
    map_points: np.ndarray
) -> np.ndarray:
    """
    Find intersections using precomputed map tangents for improved parallelism detection.

    For each trajectory point, shoots a normal ray and finds intersections with infinite
    lines defined by map points and their precomputed tangent directions. Selects the
    intersection where map and trajectory tangents are most parallel.

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array.
    tangents_map : np.ndarray
        Array of shape (M, 2) containing precomputed normalized tangent vectors
        for each map point.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing intersection points. Points with no
        valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1]])
    >>> tangents = np.array([[1, 0], [1, 0], [1, 0]])  # Horizontal tangents
    >>> knn_idx = [[0], [1], [2]]
    >>> result = find_interception_normal_shooting_parallel_tangent(
    ...     trajectory, knn_idx, tangents, map_pts)
    >>> result.shape
    (3, 2)

    Notes
    -----
    Unlike other methods that use segments, this uses infinite lines defined by
    map points and their tangent directions. This can be more robust when map
    segments are short or when the intersection might fall outside segment bounds.
    Requires parallel_score > 0.7 for a valid match.
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

        best_parallel = -np.inf  # We want the most parallel (largest dot product)
        best_intercept_point = np.nan

        knn_idx = knn_indices[i]
        for j in range(len(knn_idx)):
            map_idx = knn_idx[j]

            a = map_points[map_idx]
            tangent_map_unit = tangents_map[map_idx]  # Already normalized

            # Find intersection of normal_traj at p with the line at a with direction tangent_map_unit
            # Solve: a + t * tangent_map_unit = p + s * normal_traj
            # => t * tangent_map_unit - s * normal_traj = (p - a)
            # => [tangent_map_unit, -normal_traj] @ [t, s] = (p - a)
            A = np.column_stack((tangent_map_unit, -normal_traj))
            det = np.linalg.det(A)
            if abs(det) > 1e-10:
                sol = np.linalg.inv(A) @ (p - a)
                t, s = sol[0], sol[1]
                # No need to check t or s range, since lines are infinite
                # Compute parallelism (dot product) between tangent_map_unit and tangent_traj
                parallel = np.abs(np.dot(tangent_map_unit, tangent_traj))
                if parallel > best_parallel and parallel>0.7:
                    best_parallel = parallel
                    proj = a + t * tangent_map_unit
                    best_intercept_point = proj

        intercept_points[i] = best_intercept_point

    return intercept_points

def find_interception_normal_shooting_parallel(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    map_points: np.ndarray
) -> np.ndarray:
    """
    Find intersections prioritizing segments most perpendicular to trajectory normals.

    For each trajectory point, shoots a normal ray and finds intersections with segments
    formed by KNN neighbors. Selects the intersection where the segment direction is
    most perpendicular to the trajectory normal (most parallel to the segment).

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points used to form line segments.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing intersection points. Points with no
        valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1], [0, -1], [1, -1]])
    >>> knn_idx = [[0, 1], [1, 2], [2, 1]]
    >>> intersections = find_interception_normal_shooting_parallel(trajectory, knn_idx, map_pts)
    >>> intersections.shape
    (3, 2)

    Notes
    -----
    This variant looks for segments that are most perpendicular to the normal
    (smallest absolute dot product with normal). Requires parallel_score < 0.5
    for acceptance, meaning the segment should be reasonably perpendicular to the normal.
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

        min_parallel = np.inf
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
                    sol = np.linalg.inv(A) @ (p - a)
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        # Compute how parallel ab is to normal (smaller abs(dot) means more parallel)
                        ab_unit = ab / (np.linalg.norm(ab) + 1e-12)
                        parallel_score = abs(np.dot(ab_unit, normal))
                        if parallel_score < min_parallel:
                            min_parallel = parallel_score
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
                    sol = np.linalg.inv(A) @ (p - a)
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        ab_unit = ab / (np.linalg.norm(ab) + 1e-12)
                        parallel_score = abs(np.dot(ab_unit, normal))
                        if parallel_score < min_parallel and parallel_score<0.5:
                            min_parallel = parallel_score
                            proj = a + t * ab
                            best_intercept_point = proj

        intercept_points[i] = best_intercept_point

    return intercept_points

def find_interception_normal_shooting_nn(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    map_points: np.ndarray
) -> np.ndarray:
    """
    Find closest intersections using all pairwise segments between KNN neighbors.

    For each trajectory point, creates line segments between all pairs of its
    K-nearest neighbors and finds the closest intersection with the trajectory normal.
    This approach can be more robust when individual map segments are unreliable.

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing the closest intersection points for each
        trajectory point. Points with no valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1], [0, -1], [1, -1]])
    >>> knn_idx = [[0, 1, 3], [1, 2, 4], [2, 1, 4]]
    >>> intersections = find_interception_normal_shooting_nn(trajectory, knn_idx, map_pts)
    >>> intersections.shape
    (3, 2)

    Notes
    -----
    This method tests all possible segments formed by pairs of KNN neighbors,
    potentially providing more intersection candidates than methods that only
    use consecutive map points. Uses simple distance minimization for selection.
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
        # else:
        #     tangent = np.array([1.0, 0.0])
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular to tangent

        min_dist = np.inf
        best_intercept_point = np.nan

        knn_idx = knn_indices[i]
        
        # Work directly with NN points - create segments between all pairs
        nn_points = [map_points[idx] for idx in knn_idx]
        
        # Create line segments between all pairs of NN points
        for j in range(len(nn_points)):
            for k in range(j + 1, len(nn_points)):
                a = nn_points[j]
                b = nn_points[k]
                ab = b - a
                
                # Skip if points are too close (degenerate segment)
                if np.linalg.norm(ab) < 1e-10:
                    continue
                    
                A = np.column_stack((ab, -normal))
                det = np.linalg.det(A)
                if abs(det) > 1e-10:
                    sol = np.linalg.solve(A, p - a)
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        dist = abs(s)
                        if dist < min_dist:
                            min_dist = dist
                            proj = a + t * ab
                            best_intercept_point = proj

        intercept_points[i] = best_intercept_point

    return intercept_points

def find_interception_normal_shooting_parallel_nn(
    trajectory_points: np.ndarray, 
    knn_indices: List[List[int]], 
    map_points: np.ndarray
) -> np.ndarray:
    """
    Find intersections using sorted KNN neighbors with perpendicularity prioritization.

    For each trajectory point, sorts its KNN neighbors and creates consecutive segments,
    then finds intersections where segments are most perpendicular to the trajectory
    normal. This combines neighbor ordering with geometric constraints.

    Parameters
    ----------
    trajectory_points : np.ndarray
        Array of shape (N, 2) containing 2D trajectory points in [x, y] format.
    knn_indices : list of list of int
        For each trajectory point, a list containing indices of its K nearest
        neighbors in the map_points array.
    map_points : np.ndarray
        Array of shape (M, 2) containing 2D map points.

    Returns
    -------
    intercept_points : np.ndarray
        Array of shape (N, 2) containing intersection points. Points with no
        valid intersection are set to NaN.

    Examples
    --------
    >>> trajectory = np.array([[0, 0], [1, 0], [2, 0]])
    >>> map_pts = np.array([[0, 1], [1, 1], [2, 1], [0, -1], [1, -1]])
    >>> knn_idx = [[0, 1, 3], [1, 2, 4], [2, 1, 4]]
    >>> intersections = find_interception_normal_shooting_parallel_nn(trajectory, knn_idx, map_pts)
    >>> intersections.shape
    (3, 2)

    Notes
    -----
    This method sorts KNN neighbors lexicographically by (x, y) coordinates before
    forming consecutive segments. This can provide more consistent segment ordering
    compared to random KNN ordering. Uses perpendicularity scoring where lower
    scores indicate better perpendicular alignment.
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

        min_parallel = np.inf
        best_intercept_point = np.nan

        knn_idx = knn_indices[i]
        if len(knn_idx) == 0:
            intercept_points[i] = np.nan
            continue
        # Work directly with NN points - create segments between all pairs
        nn_points = [map_points[idx] for idx in knn_idx]
        nn_points=np.array(nn_points)
        # INSERT_YOUR_CODE
        nn_points = nn_points[np.lexsort((nn_points[:,1], nn_points[:,0]))]
        # nn_points = sorted(nn_points, key=lambda c: (c[0]-p[0])**2 + (c[1]-p[1])**2)
        # nn_points = sorted(nn_points, key=lambda c: (c[0])**2 + (c[1])**2)

        for j in range(len(nn_points)-1):
            a = nn_points[j]
            b = nn_points[j+1]
            ab = b - a
            
            # Skip if points are too close (degenerate segment)
            if np.linalg.norm(ab) < 1e-10:
                continue
                
            A = np.column_stack((ab, -normal))
            det = np.linalg.det(A)
            if abs(det) > 1e-10:
                sol = np.linalg.solve(A, p - a)
                t, s = sol[0], sol[1]
                if 0.0 <= t <= 1.0:
                    # Compute how parallel ab is to normal (smaller abs(dot) means more parallel)
                    ab_unit = ab / (np.linalg.norm(ab) + 1e-12)
                    parallel_score = abs(np.dot(ab_unit, normal))
                    if parallel_score < min_parallel:
                        min_parallel = parallel_score
                        proj = a + t * ab
                        best_intercept_point = proj

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
    max_iterations: int = 10000, 
    distance_threshold: float = 2.0, 
    inlier_ratio_threshold: float = 0.4, 
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
    
    for iteration in range(max_iterations):
        # Randomly sample points
        # Sample indices with higher probability for indices closer to N (favoring recent points)
        weights = np.linspace(0.1, 1.0, N)
        weights /= weights.sum()
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
            
        
          
    # If no good model found, fall back to standard ICP
    if best_inlier_count == 0:
        # print("Warning: RANSAC failed to find good model, falling back to standard ICP")
        best_R, best_t, final_error =  solve_trimmed_icp_2d(source_points, target_points, max_iterations=max_icp_iterations, tolerance=tolerance)
    
    # # Calculate final error using best inliers
    final_transformed_source = (best_R @ source_points.T).T + best_t
    final_error = np.mean(np.linalg.norm(final_transformed_source[best_inliers] - target_points[best_inliers], axis=1))
    
    return best_R, best_t, final_error

def transform_pose(pose: Pose, transform_matrix: np.ndarray):
    """
    Apply a 4x4 homogeneous transformation to a ROS geometry_msgs/Pose.

    Transforms both the position and orientation of a pose using a homogeneous
    transformation matrix. This is useful for converting poses between different
    coordinate frames (e.g., base_link to velodyne).

    Parameters
    ----------
    pose : geometry_msgs.msg.Pose
        ROS Pose message containing position (x, y, z) and orientation
        (quaternion: x, y, z, w) to be transformed.
    transform_matrix : np.ndarray
        Array of shape (4, 4) containing homogeneous transformation matrix
        with rotation (3x3) and translation (3x1) components.

    Returns
    -------
    transformed_pose : geometry_msgs.msg.Pose
        New ROS Pose message with transformed position and orientation.

    Examples
    --------
    >>> from geometry_msgs.msg import Pose
    >>> import numpy as np
    >>> 
    >>> # Create a pose at origin
    >>> pose = Pose()
    >>> pose.position.x, pose.position.y, pose.position.z = 1.0, 2.0, 3.0
    >>> pose.orientation.w = 1.0  # Identity rotation
    >>> 
    >>> # Translation transform
    >>> T = np.eye(4)
    >>> T[:3, 3] = [10, 20, 30]
    >>> 
    >>> transformed = transform_pose(pose, T)
    >>> print(f"New position: ({transformed.position.x}, {transformed.position.y}, {transformed.position.z})")
    New position: (11.0, 22.0, 33.0)

    Notes
    -----
    The transformation preserves the ROS message structure while applying the
    mathematical transformation. Both position and orientation are transformed
    according to the provided transformation matrix.
    
    The function handles quaternion conversions internally using scipy's
    Rotation class for robust orientation transformations.
    """
    from geometry_msgs.msg import Pose

    # Extract position as homogeneous coordinates
    position = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])
    
    # Extract orientation as quaternion and convert to rotation matrix
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rotation = Rotation.from_quat(quat)
    pose_rotation_matrix = rotation.as_matrix()
    
    # Create pose transformation matrix
    pose_transform = np.eye(4)
    pose_transform[:3, :3] = pose_rotation_matrix
    pose_transform[:3, 3] = position[:3]
    
    # Apply transformation
    transformed_matrix = transform_matrix @ pose_transform
    
    # Extract transformed position
    transformed_position = transformed_matrix[:3, 3]
    
    # Extract transformed rotation
    transformed_rotation_matrix = transformed_matrix[:3, :3]
    transformed_rotation = Rotation.from_matrix(transformed_rotation_matrix)
    transformed_quat = transformed_rotation.as_quat()
    
    # Create new Pose object
    transformed_pose = Pose()
    transformed_pose.position.x = float(transformed_position[0])
    transformed_pose.position.y = float(transformed_position[1])
    transformed_pose.position.z = float(transformed_position[2])
    transformed_pose.orientation.x = float(transformed_quat[0])
    transformed_pose.orientation.y = float(transformed_quat[1])
    transformed_pose.orientation.z = float(transformed_quat[2])
    transformed_pose.orientation.w = float(transformed_quat[3])
    
    return transformed_pose

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
