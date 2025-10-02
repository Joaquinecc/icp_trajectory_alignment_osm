#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script extracts GPS coordinates from a rosbag2 and writes them to a GeoJSON file.
It also prints the bounding box of the trajectory and the same expanded by a margin.
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Message type (ensure package is built/available in your environment)
# from inertiallabs_msgs.msg import InsData  # Not strictly needed to import directly


def meters_to_deg_lat(meters: float) -> float:
    """
    Approximate conversion from meters to degrees latitude.

    Parameters
    ----------
    meters : float
        Distance in meters.

    Returns
    -------
    float
        Distance in degrees.

    Notes
    -----
    This is an approximation and may not be accurate for large distances.
    """
    # 1 degree latitude ≈ 111,320 m (varies slightly with latitude; this is fine for ~100 m margin)
    return meters / 111320.0


def meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    """
    Approximate conversion from meters to degrees longitude at a given latitude.

    Parameters
    ----------
    meters : float
        Distance in meters.
    lat_deg : float
        Latitude in degrees.

    Returns
    -------
    float
        Distance in degrees.

    Notes
    -----
    This is an approximation and may not be accurate for large distances.
    """
    # 1 degree longitude ≈ 111,320 * cos(latitude) m
    return meters / (111320.0 * max(1e-12, math.cos(math.radians(lat_deg))))


def read_insdata_coords_from_bag(
    bag_path: str,
    topic_name: str,
    expected_ros_type: str = "inertiallabs_msgs/msg/InsData",
) -> List[Tuple[float, float]]:
    """
    Read a rosbag2 and extract (lon, lat) tuples from InsData.llh (x=lat, y=lon).
    Returns coordinates in GeoJSON order [lon, lat].

    Parameters
    ----------
    bag_path : str
        Path to the rosbag2.
    topic_name : str
        Name of the topic to read from the rosbag2.
    expected_ros_type : str, optional
        Expected ROS type of the topic.

    Returns
    -------
    coords : List[Tuple[float, float]]
        List of (lon, lat) tuples.
    """
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path not found: {bag_path}")

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic_name not in type_map:
        available = ", ".join(sorted(type_map.keys()))
        raise RuntimeError(
            f"Topic '{topic_name}' not found in bag. Available topics:\n{available}"
        )

    if type_map[topic_name] != expected_ros_type:
        print(
            f"Warning: Topic '{topic_name}' type is '{type_map[topic_name]}', "
            f"expected '{expected_ros_type}'. Attempting to deserialize anyway.",
            file=sys.stderr,
        )

    msg_cls = get_message(type_map[topic_name])

    coords: List[Tuple[float, float]] = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_cls)

        # Expect msg.llh.x (lat), msg.llh.y (lon), msg.llh.z (alt)
        try:
            lat = float(msg.llh.x)
            lon = float(msg.llh.y)
            alt = float(msg.llh.z)
        except Exception as e:
            # If structure differs, raise a clear error
            raise AttributeError(
                "Message does not contain expected fields 'msg.llh.x' and 'msg.llh.y'."
            ) from e

        # GeoJSON expects [lon, lat]
        coords.append((lon, lat, alt))

    return coords


def write_geojson_line(coords_lon_lat: List[Tuple[float, float]], out_path: str) -> None:
    """
    Write a GeoJSON LineString from a list of (lon, lat).

    Parameters
    ----------
    coords_lon_lat : List[Tuple[float, float]]
        List of (lon, lat) tuples.
    out_path : str
        Path to the output GeoJSON file.

    Returns
    -------
    None
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat, alt] for (lon, lat, alt) in coords_lon_lat],
                },
                "properties": {
                    "name": "GPS trajectory",
                    "stroke": "#ff0000",
                    "stroke-width": 3

                },
            }
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


def print_bbox_info(coords_lon_lat: List[Tuple[float, float]], margin_m: float = 100.0) -> None:
    """
    Compute and print diagonal (min/max lat/lon) and the same expanded by margin_m on each side.
    
    Parameters
    ----------
    coords_lon_lat : List[Tuple[float, float]]
        List of (lon, lat) tuples.
    margin_m : float, optional
        Margin in meters for expanded bounding box (default: 100).

    Returns
    -------
    None

    Notes
    -----
    The bounding box is printed in the format:
    min_lat, min_lon, max_lat, max_lon

    """
    lons = [c[0] for c in coords_lon_lat]
    lats = [c[1] for c in coords_lon_lat]

    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    # Print original diagonal corners
    print("\nBounding box (no margin):")
    print(f"  min corner (lon, lat): [{min_lon:.12f}, {min_lat:.12f}]")
    print(f"  max corner (lon, lat): [{max_lon:.12f}, {max_lat:.12f}]")

    # Compute margins in degrees (lon depends on latitude; use mean latitude as reference)
    mean_lat = (min_lat + max_lat) / 2.0
    dlat = meters_to_deg_lat(margin_m)
    dlon = meters_to_deg_lon(margin_m, mean_lat)

    min_lon_m = min_lon - dlon
    max_lon_m = max_lon + dlon
    min_lat_m = min_lat - dlat
    max_lat_m = max_lat + dlat

    print(f"\nBounding box (+/- {int(margin_m)} m on each side):")
    print(f"  min corner (lat, lon): [{min_lat_m:.12f}, {min_lon_m:.12f}]")
    print(f"  max corner (lat, lon): [{max_lat_m:.12f}, {max_lon_m:.12f}]")

    print(f"Easy copy paste: {min_lat_m:.12f}, {min_lon_m:.12f}, {max_lat_m:.12f}, {max_lon_m:.12f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract InsData (lat/lon) from rosbag2 and write GeoJSON LineString."
    )
    parser.add_argument("--bag", "-b", help="Path to rosbag2 (directory or .db3 file URI).", required=True)
    parser.add_argument(
        "--topic", "-t",
        required=False,
        default="/Inertial_Labs/ins_data",
        help="Topic name carrying inertiallabs_msgs/msg/InsData (e.g., /Inertial_Labs/ins_data)"
    )
    parser.add_argument(
        "--output", "-o",
        default="track.geojson",
        help="Output GeoJSON path (default: track.geojson)"
    )
    parser.add_argument(
        "--margin", "-m",
        type=float,
        default=100.0,
        help="Margin in meters for expanded bounding box (default: 100)"
    )
    args = parser.parse_args()

    coords = read_insdata_coords_from_bag(args.bag, args.topic)
    if not coords:
        print("No messages found on the specified topic. Nothing to write.", file=sys.stderr)
        sys.exit(2)

    # Write GeoJSON
    write_geojson_line(coords, args.output)

    # Print bbox info (diagonal + margin)
    print_bbox_info(coords, margin_m=args.margin)

    print(f"\nWrote GeoJSON with {len(coords)} points to: {args.output}")


if __name__ == "__main__":
    main()
