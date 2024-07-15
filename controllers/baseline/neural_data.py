#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tensorflow as tf
import random
import numpy as np
import time
import os
import pickle
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Initialize ROS parameters, publishers, subscribers, and other variables
        self.init_ros()

        # Initialize the neural network model
        self.init_neural_network()

        # Initialize history and rewards
        self.history = []
        self.rewards = []
        self.total_rewards = 0.0
        self.total_distance = 0.0
        self.previous_position = None
        self.episode = []

        # Flag to ensure initial data is received before processing starts
        self.initial_data_received = False

        # Counter for the time the robot is not upright
        self.not_upright_counter = 0
        self.upright_threshold = 10 / 0.2  # 10 seconds, assuming publish_twist is called every 0.2 seconds

    def init_ros(self):
        # Access the robot_id parameter
        self.declare_parameter('robot_id', 1)  # Default value is 1 if not set
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value

        # Create a publisher for the cmd_vel topic
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.2, self.publish_twist)

        # Initialize velocities
        self.linear_x = 0.0
        self.angular_z = 0.0

        # Initialize storage for sensor data
        self.laser_data = []
        self.odom_data_position = [0.0, 0.0, 0.0]
        self.odom_data_orientation = [0.0, 0.0, 0.0, 0.0]
        self.odom_data_linear_velocity = [0.0, 0.0, 0.0]
        self.odom_data_angular_velocity = [0.0, 0.0, 0.0]

        # Create subscribers for laser scan and odometry topics
        self.lidar_sub = self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)

    def init_neural_network(self):
        model_path = f'robot_{self.robot_id}_model.h5'
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.get_logger().info("Neural network model loaded from file.")
        else:
            # Define the neural network architecture
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2)  # 2 output nodes for linear_velocity and angular_velocity
            ])
            # Compile the model with an optimizer and a loss function (not used in RL directly)
            self.model.compile(optimizer='adam', loss='mse')
            self.get_logger().info("New neural network model created.")

        rewards_path = f'robot_{self.robot_id}_rewards.pkl'
        if os.path.exists(rewards_path):
            with open(rewards_path, 'rb') as f:
                self.total_rewards = pickle.load(f)
            self.get_logger().info("Rewards loaded from file.")
        else:
            self.total_rewards = 0.0

    def laser_callback(self, msg):
        ranges = msg.ranges
        self.laser_data = [r if np.isfinite(r) else 10.0 for r in ranges]

    def odom_callback(self, msg):
        try:
            # Update position
            self.odom_data_position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]

            # Update orientation
            self.odom_data_orientation = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]

            # Update linear velocity
            self.odom_data_linear_velocity = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ]

            # Update angular velocity
            self.odom_data_angular_velocity = [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]

            if not self.initial_data_received:
                self.previous_position = self.odom_data_position.copy()
                self.initial_data_received = True
            else:
                self.update_total_distance()

        except Exception as e:
            self.get_logger().error(f"Error in odom_callback: {e}")

    def get_input_data(self):
        return [
            *self.laser_data,
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.odom_data_linear_velocity,
            *self.odom_data_angular_velocity
        ]

    def neural_network_action(self, input_data):
        input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
        output = self.model(input_tensor)
        output_values = output.numpy()[0]

        # Handle NaN values
        if not np.isfinite(output_values).all():
            self.get_logger().warning("NaN detected in neural network output. Setting velocities to 0.")
            return 0.0, 0.0

        linear_x = float(output_values[0]) if not tf.math.is_nan(output_values[0]) else 0.0
        angular_z = float(output_values[1]) if not tf.math.is_nan(output_values[1]) else 0.0

        # Add random noise
        linear_x += random.uniform(-0.05, 0.05)
        angular_z += random.uniform(-0.05, 0.05)

        return linear_x, angular_z

    def reward_function(self):
        reward = 0.0

        # Total Distance Traveled Reward (strongest influencer)
        reward += 100.0 * self.total_distance

        # Moving Forward Reward
        if self.linear_x > 0.0:
            reward += 10.0 * abs(self.linear_x)

        # Upright Reward
        dot_product = np.dot(self.quaternion_to_up_vector(self.odom_data_orientation), [0, 0, 1])
        if dot_product > 0.8:
            reward += 5.0 * dot_product
        else:
            reward -= 10.0

        # Too Close to Object Penalty
        if self.laser_data:
            min_distance = min(self.laser_data)
            if (min_distance < 0.2):
                reward -= 1.0

        return reward

    def publish_twist(self):
        if not self.initial_data_received:
            self.get_logger().info("Publish Twist: Waiting for initial data...")
            return

        # Check if the robot is upright
        dot_product = np.dot(self.quaternion_to_up_vector(self.odom_data_orientation), [0, 0, 1])
        if dot_product > 0.8:
            self.not_upright_counter = 0
        else:
            self.not_upright_counter += 1

        # Reorient the robot if it has been not upright for more than the threshold
        if self.not_upright_counter > self.upright_threshold:
            self.get_logger().info("Robot not upright for 10 seconds, reorienting...")
            self.reorient_robot()
            self.not_upright_counter = 0
            return

        msg_cmd = Twist()
        input_data = self.get_input_data()

        # Ensure input data size matches expected input shape
        if len(input_data) != self.model.input_shape[1]:
            self.get_logger().error(f"Input data size {len(input_data)} does not match expected input shape {self.model.input_shape[1]}.")
            return

        self.linear_x, self.angular_z = self.neural_network_action(input_data)
        msg_cmd.linear.x = self.linear_x
        msg_cmd.angular.z = self.angular_z
        self.cmd_vel_pub.publish(msg_cmd)

        # Log history and rewards for potential training
        self.history.append(input_data)
        reward = self.reward_function()
        self.rewards.append(reward)
        self.total_rewards += reward

        if len(self.history) > 100:
            self.train_neural_network()

    def reorient_robot(self):
        self.get_logger().info("Reorienting the robot...")
        client = self.create_client(SpawnEntity, '/spawn_entity')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        request = SpawnEntity.Request()
        request.name = f'Robot{self.robot_id}'
        
        pkg_name = 'lp_neural_swarm'
        file_subpath = 'urdf/differential_drive_cam_lidar_fisheyes.xacro'
          
        request.xml = os.path.join(get_package_share_directory(pkg_name), file_subpath)        
        request.robot_namespace = f'/Robot{self.robot_id}'
        request.initial_pose.position.x = 0.0
        request.initial_pose.position.y = 0.0
        request.initial_pose.position.z = 0.0
        request.initial_pose.orientation.x = 0.0
        request.initial_pose.orientation.y = 0.0
        request.initial_pose.orientation.z = 0.0
        request.initial_pose.orientation.w = 1.0
        request.reference_frame = ''
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'Reoriented robot {self.robot_id}')
        else:
            self.get_logger().error('Failed to reorient the robot')

    def train_neural_network(self):
        if not self.initial_data_received:
            self.get_logger().info("Neural Network: Waiting for initial data to train neural network...")
            return

        # Convert history and rewards to numpy arrays for training
        X_train = np.array(self.history)
        y_train = np.array(self.rewards)

        # Train the model on the collected data
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Clear history and rewards for next training cycle
        self.history = []
        self.rewards = []
        self.get_logger().info("Neural network Updated.")
        
        # Save the model and total rewards
        self.model.save(f'robot_{self.robot_id}_model.h5')
        with open(f'robot_{self.robot_id}_rewards.pkl', 'wb') as f:
            pickle.dump(self.total_rewards, f)
        self.get_logger().info("Model and rewards saved.")

    def update_total_distance(self):
        if self.previous_position:
            distance = np.linalg.norm(np.array(self.odom_data_position) - np.array(self.previous_position))
            self.total_distance += distance
            self.previous_position = self.odom_data_position.copy()

    def quaternion_to_up_vector(self, q):
        x, y, z, w = q
        up_vector = np.array([
            2 * (x * y - z * w),
            1 - 2 * (x**2 + z**2),
            2 * (y * z + x * w)
        ])
        return up_vector

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.train_neural_network()  # Train the neural network after shutdown
    rclpy.shutdown()

    # Load rewards of all robots to determine the top 3
    all_rewards = {}
    for i in range(1, 11):  # Assuming there are 10 robots
        rewards_path = f'robot_{i}_rewards.pkl'
        if os.path.exists(rewards_path):
            with open(rewards_path, 'rb') as f:
                all_rewards[i] = pickle.load(f)
    
    top_robots = sorted(all_rewards.items(), key=lambda item: item[1], reverse=True)[:3]
    print(f"Top 3 robots based on total rewards: {top_robots}")

if __name__ == '__main__':
    main()
