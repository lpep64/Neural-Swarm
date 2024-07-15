#!/usr/bin/env python3
import os
import math
import random
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def spawn_coordinates_generator(num_robots, spawn_radius):
    num_points = num_robots
    spawn_coordinates = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, spawn_radius) 
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        if (x, y) not in spawn_coordinates:
            spawn_coordinates.append((x, y))
        else:
            num_points += 1
    return spawn_coordinates

def generate_launch_description():
    # Specify the name of the package and path to xacro file within the package
    pkg_name = 'lp_neural_swarm'
    file_subpath = 'urdf/differential_drive_cam_lidar_fisheyes.xacro'
    
    # Include Gazebo launch file with custom world
    world_file_name = 'cross.world'
    world_file_path = os.path.join(get_package_share_directory(pkg_name), 'worlds', world_file_name)
        
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_file_path}.items()
    )

    # Use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name), file_subpath)
    
    num_robots = 10
    spawn_radius = 3
    
    spawn_coordinates = spawn_coordinates_generator(num_robots, spawn_radius)
    
    # Generate robot descriptions
    robot_description = []
    for i in range(num_robots):
        try:
            robot_description.append(xacro.process_file(
                xacro_file, mappings={'robot_namespace': f'Robot{i+1}', 'with_ydlidar': 'true', 'yl_visualize': 'false'}
            ).toxml())
        except Exception as e:
            print(f"Error processing xacro file: {e}")
            raise
    
    # Configure the node for the robot
    node_robot_state_publisher = []
    for i in range(num_robots):
        node_robot_state_publisher.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=f'Robot{i+1}',
            output='screen',
            parameters=[{'robot_description': robot_description[i], 'use_sim_time': True}]
        ))

    # Define the spawn_entity nodes for both robots
    spawn_entities = []
    for i in range(num_robots):
        spawn_entities.append(Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name= f'spawn_entity{i+1}',
            namespace= f'Robot{i+1}',
            arguments=[
                '-topic', 'robot_description',
                '-entity', f'Robot{i+1}',
                '-robot_namespace', f'Robot{i+1}',
                '-x', str(spawn_coordinates[i][0]),
                '-y', str(spawn_coordinates[i][1]),
                '-z', '0.0',
            ],
            output='screen'
    ))
        
    # Define the controller nodes for both robots
    controller_entities = []
    for i in range(num_robots):
        controller_entities.append(Node(
            package='lp_neural_swarm',
            executable='neural_slice.py',
            name= f'neural_controller{i+1}',
            namespace= f'Robot{i+1}',
            output='screen',
            parameters=[{'robot_id': i+1}, {'total_robots': num_robots}, {'swarm_distance_threshold': 2.0}]
        ))

    # Return the LaunchDescription with both robots and nodes
    msg = [gazebo]
    for k in node_robot_state_publisher:
        msg.append(k)
    for j in spawn_entities:
        msg.append(j)
    for i in controller_entities:
        msg.append(i)  
    
    return LaunchDescription(msg)

