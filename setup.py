from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'osm_align'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'tf2_ros',
        'tf_transformations',
        'transforms3d',
    ],
    zip_safe=True,
    maintainer='joaquin-distance',
    maintainer_email='joaquin@distance.tech',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_correction_node = osm_align.nodes.lane_correction_node:main',
            'ins_conversion_node = osm_align.nodes.ins_conversion_node:main',
            'odom_enu_correction_node = osm_align.nodes.odom_enu_correction_node:main',
            'odom2gps_node = osm_align.nodes.odom2gps_node:main',
        ],
    },
)
