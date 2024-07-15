# Terminal Commands for ROS2 and Gazebo Beginners

## Remove build, install, and log directories

rm -rf build install log

## Set up ROS environment and set up the environment for the installed package

source /opt/ros/foxy/setup.bash
source install/setup.bash

## Build the selected package

colcon build --packages-select lp_neural_swarm

## List the topics

ros2 topic list

## Print a specfic topic

ros2 topic echo /topic

## List the executables provided by the package

ros2 pkg executables lp_neural_swarm

## Run the script from the package

ros2 run lp_neural_swarm script.py

## Launch the "r_neural.launch.py" file from the package

ros2 launch lp_neural_swarm r_neural.launch.py

## Publish a Twist message to the "/Robot1/cmd_vel" topic circles and stop

ros2 topic pub /Robot1/cmd_vel geometry_msgs/msg/Twist "linear:
  x: 2.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 1.0"

ros2 topic pub /Robot1/cmd_vel geometry_msgs/msg/Twist "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"

## Terminate the gzclient and server process

killall -9 gzclient
killall -9 gzserver

## Respawn Robot:

ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: 'Robot1'}"

ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{name: 'Robot1', xml: '$(cat robot_description.xml)', robot_namespace: '/Robot1', initial_pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, reference_frame: 'world'}"
