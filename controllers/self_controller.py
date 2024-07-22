#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

settings = termios.tcgetattr(sys.stdin)

class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        
        self.pub = self.create_publisher(Twist, '/Robot1/cmd_vel', 20)
        self.twist = Twist()
        
        self.timer = self.create_timer(0.1, self.publish_twist)
        self.get_logger().info('Keyboard Controller Node Initialized')

        self.quit = False

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def update_twist(self, key):
        if key == 'w' or key == 's':
            if key == 'w':
                self.twist.linear.x += 0.3
            else:
                self.twist.linear.x -= 0.3
        else:
            self.twist.linear.x *= 0.9
                
        if key == 'a' or key == 'd':
            if key == 'a':
                self.twist.angular.z += 0.2
            else:
                self.twist.angular.z -= 0.2
        else:
            self.twist.angular.z *= 0.4
        
        if key == 'z':
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
         
        if key == 'q':
            self.quit = True

    def publish_twist(self):
        if self.quit:
            self.get_logger().info('Shutting down, stopping robot')
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.pub.publish(self.twist)
            rclpy.shutdown()
            sys.exit(0)
        
        key = self.getKey()
        self.update_twist(key)
        self.pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
