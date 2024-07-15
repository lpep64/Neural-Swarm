#!/usr/bin/env python3

# ROS2 Imports
import rclpy
from rclpy.node import Node

# ROS2 Sub/Pub Imports
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

# ROS2 Service Imports (For Reorientation)
from gazebo_msgs.srv import DeleteEntity, SpawnEntity

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
        
        self.declare_parameter('total_robots', 5)
        self.total_robots = self.get_parameter('total_robots').get_parameter_value().integer_value
        
        self.source_robot_id = self.robot_id
        
        self.declare_parameter('swarm_distance_threshold', 1.0)
        self.swarm_distance_threshold = self.get_parameter('swarm_distance_threshold').get_parameter_value().double_value
        
        # Initializations
        self.init_ros()
        self.init_neural_network()
        self.init_swarm()
    
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
        self.init_flag = 0
        self.initial_data_received = False
        self.angle = 0.0
        self.not_upright_counter = 0
        self.upright_threshold = 5 / 0.2
        self.previous_position = None
        self.swarm_connections = [0] * (self.total_robots)
        self.rewards = []
        self.episode = []
        
        # ROS Subscribers
        self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)
        
        # Confirm Intialization
        self.init_flag+=1
        self.get_logger().info(f"ROS2 Node Initialized. {self.init_flag} / 3 Initializations Complete.")

        
    def init_neural_network(self):
        input_size = 15 + (3 * (self.total_robots - 1))
        
        # Neural Network Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-1, input_shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2) 
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.load_neural_network_data()
        
        # Confirm Intialization
        self.init_flag+=1
        self.get_logger().info(f"Neural Network Initialized. {self.init_flag} / 3 Initializations Complete.")
    
    def init_swarm(self):
        # Swarm Data: Total Robots -1 (Excluding Self) * 101 Data per Robot & Swarm Publisher Creation 
        self.swarm_data = (3 * (self.total_robots - 1)) * [-100]
        self.swarm_pub = self.create_publisher(Float32MultiArray, f'/Robot{self.robot_id}/swarm_data', 1)
        
        # Give time for other robots to create_publisher before atempting to subscribe
        time.sleep(2)
        
        # Subscribe to all other robots' swarm data
        for i in range(1, self.total_robots + 1):
            if i != self.robot_id:
                self.create_subscription(Float32MultiArray, f'/Robot{i}/swarm_data', self.swarm_callback, 1)
        
        # Confirm Intialization
        self.init_flag+=1
        self.get_logger().info(f"Swarm Initialized. {self.init_flag} / 3 Initializations Complete.")
    
    def laser_callback(self, msg):
        # Replace Infinite Values in Laser Data with a perceived Max of 10.0
        self.laser_data = [r if np.isfinite(r) else 10.0 for r in msg.ranges]
    
    def odom_callback(self, msg):
        # Extract Data from Odometry Message
        self.odom_data_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.odom_data_orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        # Confirm Initial Data Received
        self.initial_data_received = True

    def swarm_callback(self, msg):
    # Update Swarm Data
        if self.initial_data_received:
        # Determine if the robot is within distance of the robot sending the data
            pos1 = self.odom_data_position[:3]
            pos2 = msg.data[1:4]

            # Ensure msg.data[0] is an integer
            swarm_robot_id = int(msg.data[0])
            
            if swarm_robot_id == self.robot_id:
                return

            # Create a list of all robots in the swarm
            all_robots = list(range(1, self.total_robots + 1))
            all_robots.remove(self.robot_id)

            # Calculate the Euclidean Distance between the two robots
            distance = math.sqrt((pos1[0] - pos2[0])**2 + 
                                (pos1[1] - pos2[1])**2 + 
                                (pos1[2] - pos2[2])**2)

            # Find the index of the swarm robot ID in the list of all robots
            if swarm_robot_id in all_robots:
                index = (all_robots.index(swarm_robot_id)) * 3

                # Update the Swarm Data if the robot is within the threshold distance
                if distance <= self.swarm_distance_threshold:
                    self.swarm_data[index : index + 3] = msg.data[1:4]
                    self.swarm_connections[swarm_robot_id - 1] = 1
                else:
                    self.swarm_data[index : index + 3] = [-100, -100, -100]
                    self.swarm_connections[swarm_robot_id - 1] = 0

    def get_input_data(self):
        # Publish Swarm Data
        msg_swarm = Float32MultiArray()  
        msg_swarm.data = []
        msg_swarm.data.append(self.robot_id)
        msg_swarm.data.extend(self.odom_data_position)
        self.swarm_pub.publish(msg_swarm)
        
        # Return Input Data
        return [
            self.robot_id,
            *self.laser_data,
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.swarm_data
        ]
    
    def load_neural_network_data(self):
        try:
            # Load the neural network model and training data
            directory = 'lp_neural_swarm/data/'
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
        
        linear_x *= 0.1  
        angular_z *= 0.1 

        return linear_x, angular_z

    def reward_function(self):
        reward = 0.0
        
        # Reward for moving forward
        current_position = np.array(self.odom_data_position[:2])
        
        if self.previous_position is not None:
            position_change = np.linalg.norm(current_position - self.previous_position)
            reward += 60.0
            # Penalty for small position changes
            if position_change < 0.1:
                reward -= 20.0
        
        self.previous_position = current_position
        
        # Reward for being upright
        upright_reward = 8.0
        upside_down_penalty = -8.0
        
        rotation_matrix = utils.quaternion_to_rotation_matrix(*self.odom_data_orientation)
        up_vector = utils.get_up_vector(rotation_matrix)
        self.angle = utils.calculate_inclination_angle(up_vector)
        
        if self.angle <= 45:
            reward += upright_reward - ((upright_reward - upside_down_penalty) / 45.0) * self.angle
        else:
            reward += upside_down_penalty + ((upright_reward - upside_down_penalty) / 45.0) * (90 - self.angle)
        
        # Reward for avoiding obstacles
        if self.laser_data:
            min_distance = min(self.laser_data)
            if min_distance < 0.2:
                reward -= 100.0  # Strong penalty for very close obstacles
        
        # Reward for being connected to other robots (Swarm Connectivity)
        for i in self.swarm_connections:
            if i == 1:
                reward += 70.0  # Higher reward for strong connectivity
        
        # Speed Control (not exceeding speed limit)
        if self.linear_x <= 2.0:
            reward += 40.0  # Base reward for staying within speed limit
        else:
            reward -= 20.0 * (abs(self.linear_x) - 2.0) 
        
        # Reward for self.linear_x being positive
        if self.linear_x > 0:
            reward += 40.0
        
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
            directory = 'lp_neural_swarm/data/'
            os.makedirs(directory, exist_ok=True)

            model_path = os.path.join(directory, f'model_robot{self.robot_id}.h5')
            data_path = os.path.join(directory, f'data_robot{self.robot_id}.npz')

            self.model.save(model_path)
            np.savez(data_path, X_train=np.array(self.episode), y_train=np.array(self.rewards))
        except Exception as e:
            self.get_logger().error(f"Failed to save neural network data: {e}")

    def publish_twist(self):
        # Ensure all initializations are complete before publishing
        if not self.initial_data_received and self.init_flag >= 3:
            self.get_logger().info("Publish Twist: Waiting for initializations...")
            time.sleep(1)
            return
                
        self.check_reorient_robot()

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

    def check_reorient_robot(self):
        # Check if the robot is not upright
        if self.angle > 45:
            self.not_upright_counter += 1
            if self.not_upright_counter > self.upright_threshold:
                self.reorient_robot()
                time.sleep(9999999)
        else:
            self.not_upright_counter = 0

    def reorient_robot(self):
        self.get_logger().info("Reorienting the robot...")
        
        self.destroy_publisher(self.cmd_vel_pub)
        self.destroy_publisher(self.swarm_pub)
            
        self.destroy_subscription(self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1))
        self.destroy_subscription(self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1))

        # Delete the robot
        delete_client = self.create_client(DeleteEntity, '/delete_entity')
        delete_request = DeleteEntity.Request(name=f'Robot{self.robot_id}')
        delete_future = delete_client.call_async(delete_request)
        rclpy.spin_until_future_complete(self, delete_future, timeout_sec=1)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.train_neural_network()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
