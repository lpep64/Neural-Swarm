#!/usr/bin/env python3

# ROS2 Imports
import rclpy
from rclpy.node import Node

# ROS2 Sub/Pub Imports
from geometry_msgs.msg import Twist

# General Imports
import keyboard  # Install with: pip install keyboard

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Declaration of Robot Parameters (if needed)
        self.declare_parameter('robot_id', 1)
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.2, self.publish_twist)
        
        # ROS Variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        
        # Start keyboard listener
        keyboard.on_press(self.on_key_press)
        keyboard.on_release(self.on_key_release)
        
        self.key_mapping = {
            'w': (1.0, 0.0),    # Move forward
            's': (-1.0, 0.0),   # Move backward
            'a': (0.0, 1.0),    # Turn left
            'd': (0.0, -1.0)    # Turn right
        }
        
    def on_key_press(self, event):
        key = event.name
        
        if key in self.key_mapping:
            self.linear_x, self.angular_z = self.key_mapping[key]
        
    def on_key_release(self, event):
        key = event.name
        
        if key in self.key_mapping:
            self.linear_x, self.angular_z = 0.0, 0.0
        
    def publish_twist(self):
        twist = Twist()
        twist.linear.x = self.linear_x
        twist.angular.z = self.angular_z
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
