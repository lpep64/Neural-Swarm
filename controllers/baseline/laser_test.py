#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

class RobotController(Node):
    def __init__(self):
        super().__init__(f'robot_controller')
        
        # Get the robot_id that is intialized in the launch file
        self.declare_parameter('robot_id')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        # Create a publisher that will publish to the cmd_vel topic of the robot
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.lidar_sub = self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        
        self.timer = self.create_timer(0.1, self.publish_twist)
        self.angular_z = 0.0
        self.linear_x = 0.0
    
    def laser_callback(self, msg):
        self.get_logger().info(str(msg.ranges[6]))
        
        
    def publish_twist(self):
        msg_cmd = Twist()
        
        # Set the linear and angular velocities of the robot
        msg_cmd.linear.x = self.linear_x
        msg_cmd.linear.y = 0.0
        msg_cmd.linear.z = 0.0
        msg_cmd.angular.x = 0.0
        msg_cmd.angular.y = 0.0
        msg_cmd.angular.z = self.angular_z
        
        # Publish the velocities to the cmd_vel topic of the robot
        self.cmd_vel_pub.publish(msg_cmd)
        
        # Increment the linear velocity of the robot
        self.angular_z += 0.01
        self.linear_x += 0.01  

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    rclpy.shutdown()    

if __name__ == '__main__':
    main()