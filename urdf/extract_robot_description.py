#!/usr/bin/env python3
import os
import xacro
from ament_index_python.packages import get_package_share_directory

def get_robot_description():
    pkg_name = 'lp_neural_swarm'
    file_subpath = 'urdf/differential_drive_cam_lidar_fisheyes.xacro'
    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    
    try:
        robot_description = xacro.process_file(
            xacro_file, mappings={'robot_namespace': 'Robot1', 'with_ydlidar': 'true', 'yl_visualize': 'false'}
        ).toxml()
        return robot_description
    except Exception as e:
        print(f"Error processing xacro file: {e}")
        return None

if __name__ == '__main__':
    robot_description = get_robot_description()
    if robot_description:
        print(robot_description)
