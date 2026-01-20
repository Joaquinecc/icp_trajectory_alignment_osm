# OSM Align

ROS2 package for odometry trajectory correction using OpenStreetMap lanelet data. It was tested with LiODOM and Basalt odometry method.

## Build Map

First, generate the map from OpenStreetMap data:

```bash
python script/generate_map_utils.py --name <map_name> \
  --folder_output <output_folder> \
  --bbox <south> <west> <north> <east>
```

Example:
```bash
python script/generate_map_utils.py --name karlsruhe_map_Data \
  --folder_output ~/user/folder/ \
  --bbox 60.145248751885305 24.820470292735703 60.26639152490457 25.057807708356957
```

## Run

Launch the correction node with kitti dataset:

```bash
ros2 launch osm_align liodom_velodyne.launch.py \
  map_points_filepath:=/path/to/map_points.npz \
  bag_file:=/path/to/bag_file/ > log.out
```


## Build Package

```bash
colcon build --packages-select osm_align --symlink-install
source install/setup.bash
```

## Algorithm

The system corrects odometry drift by aligning vehicle trajectories to OpenStreetMap lane centerlines using a sliding window approach with robust point matching and trimmed ICP.

### Trajectory Accumulation

The algorithm maintains a sliding window of recent poses, starting with a minimum segment size and dynamically growing up to a maximum size. Each new pose is transformed using an accumulated correction transform and added to the window. When the window exceeds the maximum size, the oldest poses are removed to maintain computational efficiency.

### Lane Point Matching

For each trajectory point, the algorithm finds the corresponding lane centerline point through a two-stage process:

1. **Spatial Query**: A KD-tree efficiently retrieves k nearest lane points as candidates.

2. **Directional Filtering**: The trajectory tangent is computed from neighboring points. Candidate lanes are filtered by requiring the dot product between the trajectory tangent and lane direction to be at least 0.95, ensuring the lane aligns with the vehicle's heading.

3. **Normal Projection**: For directionally-aligned candidates, the algorithm projects the trajectory point onto the lane segment using normal shooting. The intersection of the trajectory normal (perpendicular to the tangent) with the lane segment is computed. If the intersection parameter falls within [0, 1], the projection point is selected as the match.

4. **Consistency Caching**: Matches that have been consistently found (≥4 consecutive times) are cached and reused to maintain temporal consistency and reduce computation.

### Alignment Process

When the window reaches the dynamic segment size, alignment is triggered:

1. **Validation**: The accumulated trajectory distance must exceed a minimum threshold, and a sufficient fraction of poses must have valid lane matches.

2. **Trimmed ICP**: A trimmed ICP algorithm is applied to the valid correspondences. The algorithm removes a fraction of the largest residuals (outliers) to handle mismatches and noise.

3. **Transform Application**: If the final ICP error is below the threshold, the computed 2D rigid transform (rotation and translation) is applied to all poses in the window. The accumulated correction transform is updated by composing the new transform with the existing one.

### Adaptive Window Management

The system adapts the window size based on alignment success:
- **Success**: The dynamic segment size increases (up to maximum), allowing the window to grow for better robustness.
- **Failure**: The oldest pose is removed, shrinking the window to prevent accumulation of poor matches.
- **Reset**: After consecutive failures, the system resets to the minimum segment size, clearing accumulated errors and allowing recovery.

This adaptive approach balances accuracy and robustness, handling varying road conditions while maintaining computational efficiency.

## Pre-commit

Before creating a PR, please run pre-commit locally before doing the commit and pushing the changes. After cloning the repo, you can install the pre-commit hooks with:

```bash
pre-commit install
```

After having added all the changes with git add, you can also manually run pre-commit as follows:

```bash
pre-commit run --all-files
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE` file for details.

## Author

**Joaquin Caballero**
Email: joaquin@gmail.com
