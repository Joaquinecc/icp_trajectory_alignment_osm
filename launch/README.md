
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
# More aggressive ICP settings for dense environments
ros2 launch osm_align osm_align.launch.py \
    frame_id:=05 \
    icp_error_threshold:=1.0 \
    min_sample_size:=15 \
    valid_correspondence_threshold:=0.7
```

#### Larger Trajectory Buffer
```bash
# Use larger pose history for more robust alignment
ros2 launch osm_align osm_align.launch.py \
    frame_id:=00 \
    pose_history_size:=100 \
    min_distance_threshold:=15.0
```

#### Custom Map Path
```bash
# Use a custom map file instead of auto-constructed path
ros2 launch osm_align osm_align.launch.py \
    frame_id:=02 \
    map_lanelet_path:=/path/to/my/custom_map.osm
```

#### Custom Topics and Services
```bash
# Use different topic names for integration with other systems
ros2 launch osm_align osm_align.launch.py \
    frame_id:=02 \
    odom_topic:=/my_robot/odom \
    transform_correction_service:=/my_lio/correction
```

## Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_id` | `'00'` | KITTI sequence identifier (e.g., '00', '01', '02', etc.) |
| `map_lanelet_path` | `''` (auto-construct) | Path to OSM lanelet file |
| `pose_history_size` | `50` | Number of poses in sliding window buffer |
| `knn_neighbors` | `10` | Number of nearest neighbors for KD-tree queries |
| `valid_correspondence_threshold` | `0.6` | Minimum ratio of valid correspondences |
| `icp_error_threshold` | `1.5` | Maximum ICP error for successful alignment |
| `min_sample_size` | `10` | Minimum samples for RANSAC-ICP |
| `min_distance_threshold` | `10.0` | Minimum trajectory distance before alignment |
| `odom_topic` | `'/liodom/odom'` | Input odometry topic name |
| `transform_correction_service` | `'/liodom/transform_correction'` | Transform correction service name |

## Parameter Tuning Guidelines

### ICP Parameters
- **`icp_error_threshold`**: Lower values (0.5-1.0) for precise environments, higher (1.5-3.0) for noisy data
- **`min_sample_size`**: Increase for robust RANSAC in outlier-heavy scenarios
- **`valid_correspondence_threshold`**: Lower (0.4-0.6) for sparse maps, higher (0.7-0.8) for dense maps

### Buffer Management
- **`pose_history_size`**: Larger values (100-200) for highways, smaller (30-50) for urban areas
- **`min_distance_threshold`**: Adjust based on expected motion (5-10m for urban, 15-30m for highways)

### Spatial Search
- **`knn_neighbors`**: Increase (15-20) for complex intersections, decrease (5-8) for simple roads

## Alternative Direct Usage

You can also run the node directly with ROS2 parameter syntax:

```bash
# Run node directly with parameters
ros2 run osm_align kitti_odometry --ros-args \
    -p frame_id:=02 \
    -p pose_history_size:=75 \
    -p icp_error_threshold:=1.2
```

## Troubleshooting

### Node doesn't start
- Check that the map file exists for the specified `frame_id`
- Verify that the `odom_topic` is being published
- Ensure the `transform_correction_service` is available

### Poor alignment performance  
- Increase `pose_history_size` for more trajectory data
- Adjust `icp_error_threshold` based on expected accuracy
- Modify `knn_neighbors` for your map density

### Too many false alignments
- Increase `valid_correspondence_threshold` 
- Raise `min_sample_size` for more robust RANSAC
- Increase `min_distance_threshold` to require more motion