# Explanation of the Sub-folders of Neural-Swarm

## controllers

The `controllers` directory contains various robot movement controllers:

* **prototypes**: Inital and experimental movement controllers
* **neural_controller.py** and **neural_slice.py**: Neural network-based controllers for robots.
* **race_controller:** Advanced 2.0 version of Neural network controllers
* **self_controller:** A self controller for robot 1 using 'wasdzq' as controls on keyboard
* **utils**: Utility files supporting the neural network controllers.

For detailed explanations of the neural network controllers, refer to `documentation/neural_explanation.md`.

## documentation

The `documentation` folder contains multiple markdown files aimed at providing users with a comprehensive understanding of the project's structure and functionality.

## launch

The `launch` directory houses:

- **prototypes**: Contains rN.launch.py an earlier version of r_neural.launch.py.
- **r_neural.launch.py**: Initializes the Gazebo simulation with a specified number of robots and spawn radius.
- **race.launch.py:** Initalizes the Gazebo simulation with 4 robots encouraged to go in along a track
- **single.launch.py**: Initalizes a Gazebo simulation with 1 robot, in another terminal if self_controller.py is run the robot can be self controlled

## lp_neural_swarm

The `lp_neural_swarm` directory contains an empty initialization file required for `CMakeLists.txt` to create the package properly.

## testing

The `testing `directory contains output files of the lidar and odometry ROS2 nodes and earlier scripts of during the design process for state subscribing and publishing.

## urdf

The `urdf` directory includes virtual representations of robots and their parts used in the URI Robotics Lab simulation.

## worlds

The `worlds` directory provides a selection of 5 Gazebo world environments for use in the simulation.
