# OSM Align - Trajectory Correction using OpenStreetMap Data

A ROS2 package for real-time odometry trajectory correction using OpenStreetMap (OSM) lanelet data. This system subscribes to odometry messages, maintains a sliding window of poses, and periodically aligns the trajectory to a high-definition map using robust ICP (Iterative Closest Point) algorithms.

## Overview

OSM Align implements trajectory-to-map alignment for autonomous vehicle navigation, correcting drift in laser odometry systems using OpenStreetMap road network data. The system uses Lanelet2 library to process OSM data and publishes corrected odometry on `/osm_align/odom_aligned`.

### Key Features

- **Real-time odometry correction** using OpenStreetMap road network data
- **Robust alignment algorithms** including ICP and RANSAC for drift correction
- **Sliding window trajectory buffering** for efficient pose management
- **Works with any odometry source** that publishes `nav_msgs/Odometry` (LIODOM was used only for testing)
- **Publishes corrected odometry** on `/osm_align/odom_aligned`
- **Configurable parameters** for different environments and datasets
- **KITTI dataset compatibility** with pre-configured coordinate systems

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Odometry      │───▶│   OSM Align     │───▶│  Corrected      │
│ (any source)    │    │     Node        │    │   Odometry      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                             │
                             ▼
                      ┌─────────────────┐
                      │  OpenStreetMap  │
                      │  Lanelet Data   │
                      └─────────────────┘
```

## Requirements

### System Requirements
- **ROS2**: Tested with ROS2 Humble
- **Python**: 3.12 (compatible with other Python 3.x versions)
- **Operating System**: Linux (tested on Ubuntu)

### Dependencies

#### ROS2 Packages
- `rclpy` - ROS2 Python client library
- `nav_msgs` - Navigation message types
- `tf2_ros` - Transform library
- `tf_transformations` - Transform utilities

#### Lanelet2 Dependencies
- `lanelet2_core` - Core lanelet functionality
- `lanelet2_io` - I/O operations for lanelet maps
- `lanelet2_projection` - Map projection utilities

#### Python Libraries
- `numpy` - Numerical computing
- `scipy` - Scientific computing (spatial algorithms, transforms)
- `transforms3d` - 3D transformation utilities

## Installation

### 1. Clone the Repository
```bash
cd ~/ros2_ws/src
git clone <repository-url> osm_align
```

### 2. Install Dependencies
```bash
# Install ROS2 dependencies
sudo apt install ros-humble-lanelet2-core ros-humble-lanelet2-io ros-humble-lanelet2-projection

# Install Python dependencies
pip install numpy scipy transforms3d
```

### 3. Build the Package
```bash
cd ~/ros2_ws
colcon build --packages-select osm_align --symlink-install
source install/setup.bash
```

## Usage

### Basic Launch
```bash
ros2 launch osm_align osm_align.launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch osm_align osm_align.launch.py \
    frame_id:=02 \
    pose_segment_size:=150 \
    icp_error_threshold:=1.5 \
    odom_topic:=/your_odom_topic
```

### Launch Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_id` | `"00"` | KITTI sequence identifier for coordinate system |
| `map_lanelet_path` | `""` | Path to OSM lanelet file (auto-constructed if empty) |
| `pose_segment_size` | `150` | Number of poses in sliding window buffer |
| `knn_neighbors` | `100` | Number of nearest neighbors for spatial queries |
| `valid_correspondence_threshold` | `0.9` | Minimum ratio of valid correspondences |
| `icp_error_threshold` | `1.5` | Maximum ICP error for successful alignment |
| `trimming_ratio` | `0.4` | Trimming ratio for robust ICP |
| `min_distance_threshold` | `10.0` | Minimum trajectory distance before alignment |
| `odom_topic` | `"/liodom/odom"` | Input odometry topic name |
| `save_resuts_path` | `'/home/.../results/osm_aligned/...'` | Directory to save results (optional) |

## Topics

### Subscribed Topics
- `odom_topic` (`nav_msgs/Odometry`) - Input odometry messages (default `/liodom/odom`)

### Published Topics
- `/osm_align/odom_aligned` (`nav_msgs/Odometry`) - Corrected odometry

## Algorithm Overview

1. **Trajectory Buffering**: Maintains a sliding window of recent odometry poses
2. **Map Loading**: Loads OSM lanelet data and converts to point cloud representation
3. **Correspondence Finding**: Uses KD-tree for efficient nearest neighbor queries
4. **Normal Shooting**: Projects trajectory normals to find map intersections
5. **Robust Alignment**: Employs trimmed/RANSAC ICP for drift-resistant pose correction
6. **Publish Corrected Odometry**: Outputs corrected poses on `/osm_align/odom_aligned`

## Testing

The package has been tested with:

### Datasets
- **KITTI Odometry Dataset** - Multiple sequences (00, 01, 02, etc.)

### Odometry Systems
- **LIODOM** - Lidar Inertial Odometry via Smoothing and Mapping (for testing)
- **Any odometry source** that publishes `nav_msgs/Odometry`

## Configuration for KITTI Dataset

The package includes pre-configured coordinate systems for KITTI sequences. Each sequence has specific:
- GPS origin coordinates for map projection
- Rotation angles for coordinate frame alignment
- Optimized algorithm parameters

## Examples

### Running with KITTI Sequence 00
```bash
ros2 launch osm_align osm_align.launch.py frame_id:=00
```

### Custom OSM Map
```bash
ros2 launch osm_align osm_align.launch.py \
    map_lanelet_path:=/path/to/your/map.osm \
    frame_id:=custom
```

## Troubleshooting

### Common Issues

1. **Map Loading Errors**: Ensure OSM file contains valid lanelet data
2. **Odometry Input**: Verify the configured `odom_topic` is being published
3. **Coordinate Frames**: Check that map and odometry coordinate systems are properly aligned

### Debug Information

Enable debug logging to monitor alignment performance:
```bash
ros2 launch osm_align osm_align.launch.py --ros-args --log-level debug
```

## Contributing

Contributions are welcome! Please follow:
1. ROS2 coding standards
2. Python PEP 8 style guidelines
3. Comprehensive documentation for new features
4. Unit tests for critical functionality

## License

This project is licensed under the Apache License 2.0. See `LICENSE` file for details.

## Maintainer

**Joaquin Distance**  
Email: joaquin@distance.tech

## Acknowledgments

- Built using [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) library
- Tested with [LIODOM](https://github.com/emiliofidalgo/liodom) odometry system
- KITTI dataset for validation and testing
