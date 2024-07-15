# neural_controller.py for Robot Navigation

## Overview

The **Neural Controller for Robot Navigation** is a Python script designed to control a robot's movement using a neural network trained on sensor data. This script integrates with the Robot Operating System (ROS) to subscribe to sensor data (laser scans and odometry) and publishes velocity commands accordingly.

## Functionality

### ROS Integration

The script uses ROS nodes to subscribe to the following topics:

- `/Robot<robot_id>/scan/out`: Laser scan data for obstacle detection.
- `/Robot<robot_id>/odom`: Odometry data for robot position and orientation.

It publishes velocity commands (`Twist` messages) to `/Robot<robot_id>/cmd_vel`.

### Neural Network Integration

The core functionality involves:

1. **Initialization**:
   - Initializes ROS parameters, publishers, subscribers, and neural network model.
   - Loads pre-trained neural network weights and historical training data if available.

2. **Data Handling**:
   - Receives and processes laser scan and odometry data.
   - Prepares input data for the neural network.

3. **Neural Network Action**:
   - Uses a neural network model (defined using TensorFlow/Keras) to predict robot velocity (`linear_x` and `angular_z`) based on input sensor data.
   - Adds random noise to the neural network output for exploration.

4. **Reward Function**:
   - Calculates a reward based on robot behavior (velocity and orientation).
   - Penalizes proximity to obstacles based on laser scan data.

5. **Training**:
   - Trains the neural network periodically using historical data and rewards accumulated during operation.
   - Saves updated neural network weights and training data for future use.

6. **Robot Reorientation**:
   - Monitors the robot's orientation and reorients it if it becomes upside-down or unstable.

### Usage

To run the script, ensure the following prerequisites are met:

- ROS (Robot Operating System) setup with necessary packages installed.
- TensorFlow and other Python dependencies (specified in `requirements.txt` or similar).

The script is typically executed as a ROS node (`rosrun package_name neural_controller.py`) and interacts with a simulated or physical robot environment.

## Files and Structure

- **neural_controller.py**: Main script handling ROS integration, neural network control, and robot behavior logic.
- **lp_neural_swarm/data/**: Directory containing saved neural network models (`model_robot<robot_id>.h5`) and training data (`data_robot<robot_id>.npz`).

## Additional Notes

- **Dependencies**: Requires Python 3.x, TensorFlow, ROS Melodic/Noetic (or compatible versions), and other specified dependencies.
- **Customization**: Users can modify neural network architecture, reward function, and training parameters based on specific robot requirements and environment characteristics.

---
