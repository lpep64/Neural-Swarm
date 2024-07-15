#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tensorflow as tf
import random
import numpy as np
import math
from std_msgs.msg import Float64MultiArray

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
        self.episode = []

        # Flag to ensure initial data is received before processing starts
        self.initial_data_received = False

    def init_ros(self):
        # Access the robot_id parameter
        self.declare_parameter('robot_id', 1)  # Default value is 1 if not set
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value

        # Create a publisher for the cmd_vel topic
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer_cmd_vel = self.create_timer(0.1, self.publish_twist)
        
        # Create a publisher for swarm state
        self.state_pub = self.create_publisher(Float64MultiArray, f'/Robot{self.robot_id}/state', 1)
        self.timer_state = self.create_timer(0.1, self.publish_state)

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
        # Define the neural network architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.get_input_data()),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)  # 2 output nodes for linear_velocity and angular_velocity
        ])

        # Compile the model with an optimizer and a loss function (not used in RL directly)
        self.model.compile(optimizer='adam', loss='mse')

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

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
                self.initial_data_received = True

        except Exception as e:
            self.get_logger().error(f"Error in odom_callback: {e}")

    def find_nearby_swarm(self):
        range_threshold = 2.5
        swarm_data = []
        
        # Iterate through known robot IDs (assuming 10 robots numbered 1 to 10)
        for k in range(1, 11):
            if k == self.robot_id:
                continue  # Skip self
            try:
                # Subscribe to swarm state topic of each robot
                swarm_sub = self.create_subscription(Float64MultiArray, f'/Robot{k}/state', self.swarm_callback, 1)
                rclpy.spin_once(self)  # Process a single callback to get latest swarm state
                distance = math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(self.swarm_data[1], self.odom_data_position)]))
                
                # Add swarm state data if within range
                if range_threshold > distance:
                    swarm_data.append(self.swarm_data)
            except Exception as e:
                self.get_logger().error(f"Error subscribing to /Robot{k}/state: {e}")

        return swarm_data

    def swarm_callback(self, msg):
        self.swarm_data = msg.data

    def get_input_data(self):
        return [
            *self.laser_data[:7],
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.odom_data_linear_velocity,
            *self.odom_data_angular_velocity,
            *self.find_nearby_swarm()  # Include swarm state data in input
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
            if min_distance < 0.2:
                reward -= 1.0

        return reward

    def publish_twist(self):
        if not self.initial_data_received:
            self.get_logger().info("Publish Twist: Waiting for initial data...")
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
        self.rewards.append(self.reward_function())
        
        if len(self.history) > 100:
            self.train_neural_network()

    def publish_state(self):
        msg_state = Float64MultiArray()
        state_data = self.get_input_data
        state_data = self.get_input_data()
        msg_state.data = state_data
        self.state_pub.publish(msg_state)

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

if __name__ == '__main__':
    main()
