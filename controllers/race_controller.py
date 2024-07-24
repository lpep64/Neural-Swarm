#!/usr/bin/env python3

# ROS2 Imports
import rclpy
from rclpy.node import Node

# ROS2 Sub/Pub Imports
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# General Imports
import tensorflow as tf
import random
import numpy as np
import time
import os
import math
from collections import deque

class RobotController(Node):
    def __init__(self):
        # Initialize the Node
        super().__init__('robot_controller')
        
        # Declaration of Robot Parameters
        self.declare_parameter('robot_id', 1)
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        self.declare_parameter('source_robot_id', 1)
        self.source_robot_id = self.get_parameter('source_robot_id').get_parameter_value().integer_value
        
        # Initializations
        self.init_ros()
        self.init_neural_network()
    
    def init_ros(self):
        # ROS Movement Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.5, self.publish_twist)  # Less frequent
        
        # ROS Variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        
        # ROS Data Variables
        self.laser_data = [0.0] * 7
        self.odom_data_position = [0.0, 0.0, 0.0]
        self.odom_data_orientation = [0.0, 0.0, 0.0, 0.0]

        # ROS Confirmation Variables (Ensuring Data is Received, Orientation is Upright, Previous Position, etc.)
        self.target_coordinates = [[(-5,0), (-5,-5), (-5,0), (0,0)],            #1/12   0
                                   [(-5,-5), (-5,-10), (0, -10), (0,-5)],       #2/12   1
                                   [(0,-5), (0,-10), (5,-10), (5,-5)],          #3/12   2
                                   [(0,0), (0,-5), (5,-5), (5,0)],              #4/12   3
                                   [(5,0), (5,-5), (10,-5), (10,0)],            #5/12   4
                                   [(5,5), (5,0), (10,0), (10,5)],              #6/12   5
                                   [(0,5), (0,0), (5,0), (5,5)],                #7/12   6
                                   [(0,10), (0,5), (5,5), (5,10)],              #8/12   7
                                   [(-5,10), (-5,5), (0,5), (0,10)],            #9/12   8
                                   [(-5,5), (-5,0), (0,0), (0,5)],              #10/12  9
                                   [(-10,5), (-10,0), (-5,0), (-5,5)],          #11/12  10
                                   [(-10,0), (-10,-5), (-5,-5), (-5,0)]]        #12/12  11
        if self.robot_id == 1:
            self.current_target = 1
        elif self.robot_id == 2:
            self.current_target = 4
        elif self.robot_id == 3:
            self.current_target = 7
        elif self.robot_id == 4:
            self.current_target = 10
        
        self.initial_data_received = False
        self.angle = 0.0
        self.not_upright_counter = 0
        self.upright_threshold = 3 / 0.2
        self.previous_position = None
        self.rewards = []
        self.episode = deque(maxlen=1000)  # Experience replay buffer
        
        self.startime = time.time()
        
        # ROS Subscribers
        self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)

    def init_neural_network(self):
        input_size = 10  # 7 laser data + 3 distances (to target, to previous, to edge)
        
        # Neural Network Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-1, input_shape=(input_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
            tf.keras.layers.Dense(2, tf.keras.activations.tanh) 
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.save_directory = 'Neural-Swarm/data'
        self.load_neural_network_data()
    
    def laser_callback(self, msg):
        # Replace Infinite Values in Laser Data with a perceived Max of 10.0
        self.laser_data = [r if np.isfinite(r) else 10.0 for r in msg.ranges[:7]]
    
    def odom_callback(self, msg):
        # Extract Data from Odometry Message
        self.odom_data_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.odom_data_orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        # Confirm Initial Data Received
        self.initial_data_received = True

    def get_input_data(self):
        # Calculate additional distance metrics
        distance_to_target = self.distance_to_edge(self.odom_data_position[0:2], self.target_coordinates[self.current_target][0], self.target_coordinates[self.current_target][1])
        distance_to_previous = 0.0
        if self.previous_position:
            distance_to_previous = math.sqrt((self.odom_data_position[0] - self.previous_position[0]) ** 2 + (self.odom_data_position[1] - self.previous_position[1]) ** 2)
        
        return [
            *self.laser_data,
            distance_to_target,
            distance_to_previous,
            self.distance_to_edge(self.odom_data_position[0:2], self.target_coordinates[self.current_target][2], self.target_coordinates[self.current_target][3])
        ]
    
    def load_neural_network_data(self):
        try:
            # Load the neural network model and training data
            directory = self.save_directory
            model_path = os.path.join(directory, f'model_robot{self.source_robot_id}.h5')
            data_path = os.path.join(directory, f'data_robot{self.source_robot_id}.npz')

            if os.path.exists(model_path) and os.path.exists(data_path):
                # Load the model
                self.model = tf.keras.models.load_model(model_path)
                self.get_logger().info(f"Neural network model loaded from {model_path}.")

                # Load the training data
                data = np.load(data_path)
                self.episode.extend(data['X_train'].tolist())
                self.rewards.extend(data['y_train'].tolist())
                self.get_logger().info(f"Training data loaded from {data_path}.")
            else:
                self.get_logger().info("No existing neural network data found. Initializing a new model.")
        except Exception as e:
            self.get_logger().error(f"Failed to load neural network data: {e}")

    def neural_network_action(self, input_data):
        input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
        output = self.model(input_tensor)
        output_values = output.numpy()[0]

        if not np.isfinite(output_values).all():
            self.get_logger().warning("NaN detected in neural network output. Setting velocities to 0.")
            return 0.0, 0.0

        # Scale the outputs to desired ranges
        linear_x = float(output_values[0]) * 0.7  # Scale if necessary
        angular_z = float(output_values[1]) * 0.7  # Scale if necessary

        # Add random noise
        linear_x += random.uniform(-0.05, 0.05)
        angular_z += random.uniform(-0.05, 0.05)

        return linear_x, angular_z

    def distance_to_edge(self, point, edge_start, edge_end):
        px, py = point
        x1, y1 = edge_start
        x2, y2 = edge_end
        
        # Line segment length squared
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            # Edge start and end are the same point
            return math.dist(point, edge_start)
        
        # Projection of point onto the line segment
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        # Distance from the point to the projection
        return math.dist(point, (projection_x, projection_y))

    def calculate_reward(self):
        if not self.initial_data_received:
            return 0.0

        reward = 0.0

        # Calculate distance to target
        target = self.target_coordinates[self.current_target]
        dist_to_target = self.distance_to_edge(self.odom_data_position[0:2], target[0], target[1])

        # Exponential reward for proximity to target
        if dist_to_target <= 0.6:
            reward += 90 * math.exp(-dist_to_target)
        else:
            reward += 10.0 / (dist_to_target + 1)

        # Penalty for being too close to obstacles
        min_distance = min(self.laser_data)
        if min_distance < 0.5:
            penalty = 70.0 * (1 - min_distance / 0.5)
            reward -= penalty

        # Encourage movement with less angular velocity
        reward += 80 * (1 - abs(self.angular_z))

        # Reward for moving forward
        if self.previous_position:
            prev_x, prev_y = self.previous_position
            current_x, current_y = self.odom_data_position[0], self.odom_data_position[1]
            distance_moved = math.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)
            reward += distance_moved * 10

        # Penalty for moving away from the target
        if self.previous_position:
            prev_dist_to_target = self.distance_to_edge(self.previous_position, target[0], target[1])
            if dist_to_target > prev_dist_to_target:
                reward -= 50.0

        # Significant reward for achieving the target
        if dist_to_target < 0.1:
            reward += 200.0  # Significant positive reward

        self.previous_position = self.odom_data_position[0:2]

        return reward


    def train_neural_network(self):
        if len(self.episode) < 100:
            return
        
        # Random sample from experience replay buffer
        batch = random.sample(self.episode, 100)
        
        # Prepare training data
        X_train = np.array([experience['state'] for experience in batch])
        y_train = np.array([experience['reward'] for experience in batch])
        
        # Fit the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])
        
        # Save the model and training data
        self.model.save(os.path.join(self.save_directory, f'model_robot{self.robot_id}.h5'))
        np.savez(os.path.join(self.save_directory, f'data_robot{self.robot_id}.npz'), X_train=X_train, y_train=y_train)
        self.get_logger().info("Model and data saved.")
        
    def publish_twist(self):
        if not self.initial_data_received:
            return

        # Get the input data
        input_data = self.get_input_data()

        # Get the actions from the neural network
        self.linear_x, self.angular_z = self.neural_network_action(input_data)

        # Publish the movement command
        twist = Twist()
        twist.linear.x = self.linear_x
        twist.angular.z = self.angular_z
        self.cmd_vel_pub.publish(twist)

        # Calculate the reward
        reward = self.calculate_reward()
        self.rewards.append(reward)

        # Store the experience
        experience = {
            'state': input_data,
            'reward': reward
        }
        self.episode.append(experience)
        
        # Train the neural network less frequently
        if len(self.episode) % 100 == 0:
            self.train_neural_network()

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
