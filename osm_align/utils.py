import numpy as np
def find_interception_normal_shooting(trajectory_points, knn_indices, map_points):
    """
    For each trajectory point, shoot a normal and find the closest intersection with a segment
    formed by its KNN neighbors in the map. Returns a list of intercept points.
    Args:
        trajectory_points: np.ndarray of shape (N, 2)
        knn_indices: list of lists of int, each sublist contains indices into map_points
        map_points: np.ndarray of shape (M, 2)
    Returns:
        intercept_points: np.ndarray of shape (N, 2)
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
                    sol = np.linalg.solve(A, p - a)
                    # sol=  np.linalg.lstsq(A, p-a, rcond=None)[0]
                    t, s = sol[0], sol[1]
                    if 0.0 <= t <= 1.0:
                        dist = abs(s)
                        if dist < min_dist:
                            min_dist = dist
                            proj = a + t * ab
                            best_intercept_point = proj
                            if i ==8:
                                print(f"a {a} b {b}")
                                print(f"map_idx {map_idx}")
                                print(f"ab {ab}")
                                print(f"Intersection found at {proj}")

            # Try next neighbor
            if map_idx + 1 < len(map_points):
                a = map_points[map_idx]
                b = map_points[map_idx + 1]
                ab = b - a
                A = np.column_stack((ab, -normal))
                det = np.linalg.det(A)
                if abs(det) > 1e-10:
                    sol = np.linalg.solve(A, p - a)
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

def find_interception_normal_shooting_nn(trajectory_points, knn_indices, map_points):
    """
    For each trajectory point, shoot a normal and find the closest intersection with segments
    formed by its KNN neighbors. Works directly with NN points instead of map points.
    Args:
        trajectory_points: np.ndarray of shape (N, 2)
        knn_indices: list of lists of int, each sublist contains indices into map_points
        map_points: np.ndarray of shape (M, 2)
    Returns:
        intercept_points: np.ndarray of shape (N, 2)
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
def solveIcp2d( source, target, max_iterations=50, tolerance=1e-6):
    """
    Solves the 2D ICP problem between source and target points.

    Args:
        source (np.ndarray): Nx2 array of source points.
        target (np.ndarray): Nx2 array of target points.
        max_iterations (int): Maximum number of ICP iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        R_total (np.ndarray): 2x2 rotation matrix.
        T_total (np.ndarray): 2x1 translation vector.
        final_error (float): Mean registration error (scalar).
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
    
def solve_trimmed_icp_2d(source_points, target_points, trimming_ratio=0.1, max_iterations=50, tolerance=1e-6):
    """
    Python implementation of trimmed ICP for 2D point sets.
    Args:
        source_points: np.ndarray of shape (N, 2)
        target_points: np.ndarray of shape (N, 2)
        trimming_ratio: float in [0, 1), fraction of correspondences to trim (remove as outliers)
        max_iterations: int, maximum number of ICP iterations
        tolerance: float, convergence threshold on mean error change
    Returns:
        R_total: np.ndarray of shape (2, 2), final rotation matrix
        t_total: np.ndarray of shape (2,), final translation vector
        icp_error: float, mean error over best correspondences
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
    icp_error = np.mean(np.linalg.norm(correct_source_points - target_points, axis=1))
    return R_total, t_total, icp_error

