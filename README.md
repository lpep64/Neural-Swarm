# Repository Name

This repository contains simulations for the work done at the URI Robotics Lab, created by Professor Paolo Stegagno.

## Prerequisites

In order to properly run the simulations, you will need the following dependencies:

- ROS2 Foxy
- Gazebo
- ros-foxy-xacro
- ros-foxy-gazebo-ros-pkgs

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/pstegagno/us2_gazebo.git
    ```

2. Install the required dependencies:

    ```bash
    sudo apt install ros-foxy-gazebo-ros-pkgs ros-foxy-xacro
    ```

## Usage

1. Build the workspace:

    ```bash
    cd /root/ros2_ws/us2_gazebo_diffdrive
    colcon build
    ```

2. Launch the simulation:

    ```bash
    source /root/ros2_ws/install/setup.bash
    ros2 launch lp_neural_swarm spawn_diff_drive.launch.py

    ```

## Contributing

Created by Paolo Stegagno, if you contributing researcher please sign below for future users:

Adam Hoburg (Pre-Repository Work)

Luke Pepin (Summer 2024)