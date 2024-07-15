#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self, robot_id):
        super().__init__(f'robot_controller')
        
        # Create a publisher that will publish to the cmd_vel topic of the robot
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.publish_twist)

    def publish_twist(self):
        msg_cmd = Twist()
        
        # Set the linear and angular velocities of the robot to all zeros
        msg_cmd.linear.x = 0.0
        msg_cmd.linear.y = 0.0
        msg_cmd.linear.z = 0.0
        msg_cmd.angular.x = 0.0
        msg_cmd.angular.y = 0.0
        msg_cmd.angular.z = 0.0 
        
        # Publish the velocities to the cmd_vel topic of the robot      
        self.cmd_vel_pub.publish(msg_cmd)
       

def main(args=None):
    rclpy.init(args=args)

    # Check if a robot_id was passed as an argument
    if len(args) < 2:
        print("Usage: python3 solo_stop.py <robot_id>")
        return

    try:
        # Try to convert the robot_id to an integer
        robot_id = int(args[1])
    except ValueError:
        print("Error: robot_id must be an integer")
        return

    try:
        # Try to create a RobotController and spin it
        robot_controller = RobotController(robot_id)
        rclpy.spin(robot_controller)
    except Exception as e:
        print(f"Error: {e}")
        return

    rclpy.shutdown() 

if __name__ == '__main__':
    main()