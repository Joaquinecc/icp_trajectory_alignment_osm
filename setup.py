from setuptools import find_packages, setup

package_name = 'osm_align'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'odom_subscriber = osm_align.odometry:main',
        ],
    },
)
