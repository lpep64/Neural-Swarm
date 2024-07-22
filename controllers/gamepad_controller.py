#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class GamepadController(Node):
    def __init__(self):
        super().__init__('gamepad_controller')
        
        self.pub = self.create_publisher(Twist, '/Robot1/cmd_vel', 20)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.twist = Twist()
        self.get_logger().info('Gamepad Controller Node Initialized')

    def joy_callback(self, msg):
        # Assuming the left stick controls linear x and right stick controls angular z
        self.twist.linear.x = msg.axes[1] * 2.0  # Adjust the scale factor if needed
        self.twist.angular.z = msg.axes[3] * 2.0  # Adjust the scale factor if needed
        self.pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = GamepadController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
