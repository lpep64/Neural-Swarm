#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Specify the name of the package and path to xacro file within the package
    pkg_name = 'lp_neural_swarm'
    file_subpath = 'urdf/differential_drive_cam_lidar_fisheyes.xacro'

    # Use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    
    # Generate robot descriptions
    try:
        robot_description1 = xacro.process_file(
            xacro_file, mappings={'robot_namespace': 'Robot1', 'with_ydlidar': 'true', 'yl_visualize': 'true'}
        ).toxml()
    except Exception as e:
        print(f"Error processing xacro file: {e}")
        raise

    # Configure the first node for the first robot
    node_robot_state_publisher1 = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='Robot1',
        output='screen',
        parameters=[{'robot_description': robot_description1, 'use_sim_time': True}]
    )

    # Include Gazebo launch file
    world_file_name = 'racetrack.world'
    world_file_path = os.path.join(get_package_share_directory(pkg_name), 'worlds', world_file_name)
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_file_path}.items()
    )

    # Define the spawn_entity nodes for the robot
    spawn_entity1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity1',
        namespace='Robot1',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'Robot1',
            '-robot_namespace', 'Robot1',
            '-x', '2.5',
            '-y', '2.5',
            '-z', '0.0',
        ],
        output='screen'
    )
    
    # Define the controller node for the robot
    controller = Node(
        package='lp_neural_swarm',
        executable='self_controller.py',
        name='self_controller',
        namespace='Robot1',
        output='screen',
        parameters=[{'robot_id': 1}]
    )

    # Return the LaunchDescription with both robots and nodes
    return LaunchDescription([
        gazebo,
        node_robot_state_publisher1,
        spawn_entity1,
        controller
    ])
