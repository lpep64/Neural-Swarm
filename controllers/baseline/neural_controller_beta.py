#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry  
from sensor_msgs.msg import LaserScan
import tensorflow as tf
import time
import random
import numpy as np

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Access the robot_id parameter
        self.declare_parameter('robot_id', 1)  # Default value is 1 if not set
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        # Create a publisher for the cmd_vel topic
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.publish_twist)
        
        # Initialize the linear and angular velocities
        self.linear_x = 0.0  
        self.angular_z = 0.0
        
        # Initialize storage for sensor data
        self.laser_data = []
        self.odom_data_position = [0.0, 0.0, 0.0]
        self.odom_data_orientation = [0.0, 0.0, 0.0, 0.0]
        self.odom_data_linear_velocity = [0.0, 0.0, 0.0]
        self.odom_data_angular_velocity = [0.0, 0.0, 0.0]
        
        self.estimated_flag = 0
        self.estimated_position = [0.0, 0.0, 0.0]
        
        self.history  = []
        self.rewards = []
        self.episode = []
        
        # Create subscribers for laser scan and odometry topics
        self.lidar_sub = self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)
    
    def laser_callback(self, msg):
        self.laser_data = msg.ranges[:7]        
        # self.get_logger().info('Laser data: %s' % self.laser_data)

    def odom_callback(self, msg):
        # Actual Position [x, y, z]
        self.odom_data_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        
        # Actual Orientation [x, y, z, w]
        self.odom_data_orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        
        # Linear Velocity [x, y, z]
        self.odom_data_linear_velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]
        
        # Angular Velocity [x, y, z]
        self.odom_data_angular_velocity = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ]
        
        # Estimated Position [x, y, z]
        if self.estimated_flag == 0:
            self.estimated_position = self.odom_data_position
            self.estimated_flag = 1
        else:
            self.estimated_position = [
                self.estimated_position[0] + self.odom_data_linear_velocity[0],
                self.estimated_position[1] + self.odom_data_linear_velocity[1],
                self.estimated_position[2] + self.odom_data_linear_velocity[2]
            ]
            
        # self.get_logger().info('Odom data: %s' % self.odom_data_position[0])
    
    def NeuralNetwork(self):
        input_data = [
            *self.laser_data[:7],
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.odom_data_linear_velocity,
            *self.odom_data_angular_velocity
        ]

        # Define the neural network architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(input_data),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)  # 2 output nodes for linear_velocity and angular_velocity
        ])

        input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
        output = self.model(input_tensor)

        # Extracting values from the output tensor
        output_values = output.numpy()[0]

        # Ensure outputs are floats
        self.linear_x = float(output_values[0]) if not tf.math.is_nan(output_values[0]) else 0.0
        self.angular_z = float(output_values[1]) if not tf.math.is_nan(output_values[1]) else 0.0
        
        self.linear_x += random.uniform(-0.5, 0.5)  # Example range for linear_x noise
        self.angular_z += random.uniform(-0.5, 0.5)  # Example range for angular_z noise
                
        self.history.append(input_data)
        self.rewards.append(self.reward_function())
        self.episode.append([input_data, [self.linear_x, self.angular_z], self.rewards[-1]])
        
        #self.get_logger().info('Linear x:  %s' % self.linear_x)
        #self.get_logger().info('Angular z: %s' % self.angular_z)
    
    def reward_function(self):
        reward = 0.0
        if self.linear_x > 0.0:
            reward += 3.0
        dot_product = np.dot(quaternion_to_up_vector(self.odom_data_orientation), [0, 0, 1])
        if dot_product > 0.8:
            reward += 5.0
        return reward
        
    def publish_twist(self):
        msg_cmd = Twist()
        self.NeuralNetwork()
        
        # Set the linear and angular velocities
        msg_cmd.linear.x = self.linear_x
        msg_cmd.angular.z = self.angular_z
        
        # Publish the velocities to the cmd_vel topic
        self.cmd_vel_pub.publish(msg_cmd)

def quaternion_to_up_vector(q):
    x, y, z, w = q
    # Calculate the up vector from the quaternion
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
    rclpy.shutdown()

if __name__ == '__main__':
    main()