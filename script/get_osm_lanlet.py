"""
get_osm_lanlet.py

This script downloads OpenStreetMap (OSM) data for a given bounding box, patches the data
by converting all 'unclassified' highways to 'residential', and then converts the result
to Lanelet2 format using crdesigner.

Usage:
    python get_osm_lanlet.py --name <output_name> --folder-output <output_folder> [--bbox S W N E]

Arguments:
    --name            Name for the output files (required)
    --folder_output   Output folder where files will be saved (required)
    --bbox            Bounding box as four floats: south west north east (optional, default is Helsinki area)

Example:
    python get_osm_lanlet.py --name dummy --folder-output ./output_folder --bbox 60.17 24.94 60.18 24.96

This will:
    - Download OSM data for the bounding box (60.17, 24.94, 60.18, 24.96)
    - Replace all 'unclassified' highway tags with 'residential'
    - Save the patched OSM XML to ./output_folder/dummy_result.osm
    - Convert to CRM format and save as ./output_folder/dummy_crm.xml
    - Convert to Lanelet2 format and save as ./output_folder/dummy_lanelet2.osm

Requirements:
    - requests
    - crdesigner (must be installed and available in PATH)
"""
import argparse
import os
import requests
import xml.etree.ElementTree as ET
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Download OSM, patch, and convert to Lanelet2")
    parser.add_argument("--name", required=True, help="Name for the output files")
    parser.add_argument("--folder-output", required=True, help="Output folder")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=('S', 'W', 'N', 'E'),
                        default=[60.176021478825, 24.925027062444,60.186692008175, 24.952570024556],
                        help="Bounding box: south west north east")
    args = parser.parse_args()

    name = args.name
    folder_output = args.folder_output
    bbox = args.bbox

    os.makedirs(folder_output, exist_ok=True)
    #|service can be added to the query 
    query = f"""
    [timeout:500][out:xml][bbox:
    {bbox[0]},{bbox[1]},
    {bbox[2]},{bbox[3]}
    ];
    way
      [highway~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street)$"];
    (._;>;);
    out meta qt;
    """.strip()
    # Fetch OSM XML from Overpass
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=UTF-8"},
        timeout=200,
    )
    r.raise_for_status()

    # Parse XML
    root = ET.fromstring(r.content)

    # Iterate and replace tag values
    for tag in root.iter("tag"):
        if tag.attrib.get("k") == "highway" and tag.attrib.get("v") == "unclassified":
            tag.set("v", "residential")

    # Save updated XML
    osm_path = os.path.join(folder_output, f"{name}_result.osm")
    tree = ET.ElementTree(root)
    tree.write(osm_path, encoding="utf-8", xml_declaration=True)

    # Convert to CRM format
    crm_path = os.path.join(folder_output, f"{name}_crm.xml")
    lanelet2_path = os.path.join(folder_output, f"{name}_lanelet2.osm")

    # Run crdesigner for osmcr
    subprocess.run([
        "crdesigner",
        "--input-file", osm_path,
        "--output-file", crm_path,
        "osmcr"
    ], check=True)

    # Run crdesigner for crlanelet2
    subprocess.run([
        "crdesigner",
        "--input-file", crm_path,
        "--output-file", lanelet2_path,
        "crlanelet2"
    ], check=True)

if __name__ == "__main__":
    main()
