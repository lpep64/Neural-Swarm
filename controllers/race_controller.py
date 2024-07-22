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
import utils
import math

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
        self.timer = self.create_timer(0.2, self.publish_twist)
        
        # ROS Variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        
        # ROS Data Variables
        self.laser_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.odom_data_position = [0.0, 0.0, 0.0]
        self.odom_data_orientation = [0.0, 0.0, 0.0, 0.0]

        # ROS Confrimation Variables (Ensuring Data is Received, Orientation is Upright, Previous Position, etc.)
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
        self.episode = []
        
        self.startime = time.time()
        
        # ROS Subscribers
        self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)

        
    def init_neural_network(self):
        input_size = 16
        
        # Neural Network Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-1, input_shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2) 
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.save_directory = 'Neural-Swarm/data'
        self.load_neural_network_data()
        
    
    def laser_callback(self, msg):
        # Replace Infinite Values in Laser Data with a perceived Max of 10.0
        self.laser_data = [r if np.isfinite(r) else 10.0 for r in msg.ranges]
    
    def odom_callback(self, msg):
        # Extract Data from Odometry Message
        self.odom_data_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.odom_data_orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        # Confirm Initial Data Received
        self.initial_data_received = True

    def get_input_data(self):
        return [
            self.robot_id,
            *self.laser_data,
            *self.odom_data_position,
            *self.odom_data_orientation,
            self.distance_to_edge(self.odom_data_position[0:2], self.target_coordinates[self.current_target][0], self.target_coordinates[self.current_target][1])
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
                self.episode = data['X_train'].tolist()
                self.rewards = data['y_train'].tolist()
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

        linear_x = float(output_values[0]) if not tf.math.is_nan(output_values[0]) else 0.0
        angular_z = float(output_values[1]) if not tf.math.is_nan(output_values[1]) else 0.0
               
        linear_x += random.uniform(-0.05, 0.05)
        angular_z += random.uniform(-0.05, 0.05)
        
        if linear_x > 0.50:
            linear_x = 0.50

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

    def check_target_reached(self):
        # Extract the coordinates of the square
        x1, y1 = self.target_coordinates[self.current_target][0]  # Top left
        x2, y2 = self.target_coordinates[self.current_target][1]  # Bottom left
        x3, y3 = self.target_coordinates[self.current_target][2]  # Bottom right
        x4, y4 = self.target_coordinates[self.current_target][3]  # Top right

        # Extract the coordinates of the point
        px = self.odom_data_position[0]
        py = self.odom_data_position[1]

        # Check if the point is within the x and y bounds of the square
        if min(x1, x2, x3, x4) <= px <= max(x1, x2, x3, x4) and min(y1, y2, y3, y4) <= py <= max(y1, y2, y3, y4):
            return 0.0
        else:
            # Calculate distances to each edge
            edges = [(self.target_coordinates[self.current_target][0], self.target_coordinates[self.current_target][1]),
                    (self.target_coordinates[self.current_target][1], self.target_coordinates[self.current_target][2]),
                    (self.target_coordinates[self.current_target][2], self.target_coordinates[self.current_target][3]),
                    (self.target_coordinates[self.current_target][3], self.target_coordinates[self.current_target][0])]
            
            distances = [self.distance_to_edge((px, py), edge_start, edge_end) for edge_start, edge_end in edges]
            min_distance = min(distances)
            return min_distance
    
    def reward_function(self):
        reward = 0.0
        
        dist_to_target = self.check_target_reached()        
        if dist_to_target == 0:
            reward += 100.0
            self.current_target = (self.current_target + 1) % 12
            self.startime = time.time()
            self.get_logger().info(f"Target reached. Moving to target {self.current_target}.")
        else:
            reward += 100.0 / (dist_to_target + 1)
            
        rotation_matrix = utils.quaternion_to_rotation_matrix(*self.odom_data_orientation)
        up_vector = utils.get_up_vector(rotation_matrix)
        self.angle = utils.calculate_inclination_angle(up_vector)

        upright_reward = 30.0
        upside_down_penalty = -50.0

        if self.angle <= 45:
            reward += upright_reward - ((upright_reward - upside_down_penalty) / 90.0) * self.angle
        else:
            reward += upside_down_penalty + ((upright_reward - upside_down_penalty) / 90.0) * (180 - self.angle)

        if self.laser_data:
            min_distance = min(self.laser_data)
            if min_distance < 0.2:
                reward -= 100.0
        
        elapsed_time = time.time() - self.startime    
        reward -= 10 * (1 - np.exp(-0.1 * elapsed_time))
        
        if self.linear_x > 0.0:
            reward += 10.0
        
        return reward

    def train_neural_network(self):
        if not self.initial_data_received:
            self.get_logger().info("Neural Network: Waiting for initial data to train neural network...")
            return

        X_train = np.array(self.episode)
        y_train = np.array(self.rewards)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
        self.save_neural_network_data()

        self.episode = []
        self.rewards = []
        self.get_logger().info("Neural network updated.")

    def save_neural_network_data(self):
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)

            model_path = os.path.join(self.save_directory, f'model_robot{self.robot_id}.h5')
            data_path = os.path.join(self.save_directory, f'data_robot{self.robot_id}.npz')

            # Delete existing files if they exist
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(data_path):
                os.remove(data_path)

            # Save the model
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Save the training data (self.episode and self.rewards)
            np.savez(data_path, X_train=np.array(self.episode), y_train=np.array(self.rewards))
            print(f"Training data saved to: {data_path}")

        except Exception as e:
            print(f"Failed to save neural network data: {e}")

    def publish_twist(self):
        # Ensure all initializations are complete before publishing
        if not self.initial_data_received:
            self.get_logger().info("Publish Twist: Waiting for initializations...")
            time.sleep(1)
            return
                
        # Get Input Data and Publish Twist
        msg_cmd = Twist()
        input_data = self.get_input_data()

        if len(input_data) != self.model.input_shape[1]:
            self.get_logger().error(f"Input data size {len(input_data)} does not match expected input shape {self.model.input_shape[1]}.")         
            return
        
        self.linear_x, self.angular_z = self.neural_network_action(input_data)
        msg_cmd.linear.x = self.linear_x
        msg_cmd.angular.z = self.angular_z
        self.cmd_vel_pub.publish(msg_cmd)

        # Train Neural Network
        self.episode.append(input_data)
        self.rewards.append(self.reward_function())

        # Train Neural Network if Episode is Long
        if len(self.episode) >= 25:
            self.train_neural_network()

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.train_neural_network()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
