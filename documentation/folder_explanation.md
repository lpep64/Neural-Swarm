# Explanation of the Sub-folders of Neural-Swarm

## controllers

The `controllers` directory contains various robot movement controllers:

* **baseline**: Default movement controllers without neural networks.
* **neural_controller.py** and **neural_slice.py**: Neural network-based controllers for robots.
* **utils**: Utility files supporting the neural network controllers.

For detailed explanations of the neural network controllers, refer to `documentation/neural_explanation.md`.

## documentation

The `documentation` folder contains multiple markdown files aimed at providing users with a comprehensive understanding of the project's structure and functionality.

## launch

The `launch` directory houses:

- **r_neural_launch.py**: Initializes the Gazebo simulation with a specified number of robots and spawn radius.

## lp_neural_swarm

The `lp_neural_swarm` directory contains an empty initialization file required for `CMakeLists.txt` to create the package properly.

## urdf

The `urdf` directory includes virtual representations of robots and their parts used in the URI Robotics Lab simulation.

## worlds

The `worlds` directory provides a selection of 5 Gazebo world environments for use in the simulation.
