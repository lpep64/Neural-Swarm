# Neural-Swarm

This repository contains simulations for the work done at the URI Robotics Lab, created by Luke Pepin overseen by Professor Paolo Stegagno.

## Introduction

This project, Neural-Swarm or "Neural Network Robotic Swarms: Communication, Mapping, and Navigation," led by Luke Pepin and overseen by Professor Paolo Stegagno from the University of Connecticut and the University of Rhode Island, Department of Electrical, Computer and Biomedical Engineering, focuses on developing autonomous robot swarms capable of efficiently mapping and navigating environments. Using neural networks, the project enhances swarm decision-making under constraints on communication and sensor data. Applications include scenarios like search and rescue and environmental monitoring, implemented on ROS 2 (Foxy) and Ubuntu 20.04. The project aims to advance robotic swarm capabilities in real-world environments with limited communication, featuring robust testing and ongoing development to improve swarm efficiency and adaptability.

## Getting Started

### Prerequisites

In order to properly run the simulations and use the project on Ubuntu 20.04, you will need the following dependencies:

* ROS2 Foxy
* Gazebo
* ros-foxy-xacro
* ros-foxy-gazebo-ros-pkgs

Make sure to have these dependencies installed and configured before running the simulations or deploying the project

### Installation

1. In Ubuntu 20.04, Clone this repository:

   ```shell
   git clone https://github.com/lpep64/Neural-Swarm
   ```
2. Install the required dependencies:

   ```
   sudo apt install ros-foxy-gazebo-ros-pkgs ros-foxy-xacro
   ```

### Usage

Neural-Swarm/documentation/terminal_commands.md contains most of the nessary commands needed for usage however the following are the most important:

Set up ROS environment and set up the environment for the installed package

```
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

Build the Selected Package:

```
colcon build --packages-select lp_neural_swarm
```

Launch the "r_neural.launch.py" file from the package

```
ros2 launch lp_neural_swarm r_neural.launch.py
```

## Configuration

The primary code for running the neural-swarm is `r_neural_launch.py`, which initializes the Gazebo simulation with a default of 10 robots within a spawn radius of 2 meters. Each robot executes `neural_controller.py`, managing necessary subscribers, publishers, and the neural network itself.

## Contributing

This project serves as a reference for the work I completed during the summer of 2024 at URI for NIUVT-ANCHOR. Contributions and updates to enhance the project are encouraged.

## Acknowledgements

Thank you to DoDstem Navy Stem Crew, NIUVT, URI, and the URI Department of Electrical, Computer and Biomedical Engineering for giving me the opportunity to work on and create this project. Special thanks to Adam Hoburg for his contributions; his ROS1 code provided valuable reference material during the initial stages of development.

## Contact

For inquiries, please contact Luke Pepin via email at [lukepepin@outlook.com]().
