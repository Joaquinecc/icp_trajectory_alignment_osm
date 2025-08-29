
# OSM Alignment Launch File Usage

This document explains how to use the OSM alignment launch file to configure and run the odometry correction system with customizable parameters.

## Quick Start

### Basic Usage (Default Parameters)
```bash
# Launch with all default parameters (frame_id='00')
ros2 launch osm_align osm_align.launch.py
```

### Specify KITTI Sequence
```bash
# Launch with KITTI sequence 02
ros2 launch osm_align osm_align.launch.py frame_id:=02
```

### Custom Configuration Examples

#### Adjust ICP Parameters
```bash
# ICP settings for dense environments
ros2 launch osm_align osm_align.launch.py \
    frame_id:=05 \
    icp_error_threshold:=1.5 \
    valid_correspondence_threshold:=0.9 \
    trimming_ratio:=0.4
```

#### Larger Trajectory Buffer
```bash
# Use larger pose history for more robust alignment
ros2 launch osm_align osm_align.launch.py \
    frame_id:=00 \
    pose_segment_size:=150 \
    min_distance_threshold:=15.0
```

#### Custom Map Path
```bash
# Use a custom map file instead of auto-constructed path
ros2 launch osm_align osm_align.launch.py \
    frame_id:=02 \
    map_lanelet_path:=/path/to/my/custom_map.osm
```

#### Custom Topics
```bash
# Use a different input odometry topic
ros2 launch osm_align osm_align.launch.py \
    frame_id:=02 \
    odom_topic:=/my_robot/odom
```

## Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_id` | `'00'` | KITTI sequence identifier (e.g., '00', '01', '02', etc.) |
| `map_lanelet_path` | `''` (auto-construct) | Path to OSM lanelet file |
| `pose_segment_size` | `150` | Number of poses in sliding window buffer |
| `knn_neighbors` | `100` | Number of nearest neighbors for KD-tree queries |
| `valid_correspondence_threshold` | `0.9` | Minimum ratio of valid correspondences |
| `icp_error_threshold` | `1.5` | Maximum ICP error for successful alignment |
| `trimming_ratio` | `0.4` | Trimming ratio for robust ICP |
| `min_distance_threshold` | `10.0` | Minimum trajectory distance before alignment |
| `odom_topic` | `'/liodom/odom'` | Input odometry topic name |
| `save_resuts_path` | `'/home/.../results/osm_aligned/...'` | Directory to save results (optional) |

## Parameter Tuning Guidelines

### ICP Parameters
- **`icp_error_threshold`**: Lower values (0.5-1.0) for precise environments, higher (1.5-3.0) for noisy data
- **`valid_correspondence_threshold`**: Lower (0.6-0.8) for sparse maps, higher (0.9) for dense maps
- **`trimming_ratio`**: Increase to be more conservative against outliers; decrease if losing inliers

### Buffer Management
- **`pose_segment_size`**: Larger values (100-200) for highways, smaller (30-80) for urban areas
- **`min_distance_threshold`**: Adjust based on expected motion (5-10m for urban, 15-30m for highways)

### Spatial Search
- **`knn_neighbors`**: Increase (15-20) for complex intersections, decrease (5-10) for simple roads

## Alternative Direct Usage

You can also run the node directly with ROS2 parameter syntax:

```bash
# Run node directly with parameters
ros2 run osm_align kitti_odometry --ros-args \
    -p frame_id:=02 \
    -p pose_segment_size:=150 \
    -p icp_error_threshold:=1.5
```

## Troubleshooting

### Node doesn't start
- Check that the map file exists for the specified `frame_id`
- Verify that the `odom_topic` is being published

### Poor alignment performance  
- Increase `pose_segment_size` for more trajectory data
- Adjust `icp_error_threshold` based on expected accuracy
- Modify `knn_neighbors` for your map density

### Too many false alignments
- Increase `valid_correspondence_threshold` 
- Increase `min_distance_threshold` to require more motion