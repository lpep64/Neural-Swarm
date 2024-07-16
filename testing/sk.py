#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import threading
import time

def get_laser_data(robot_id):
    # Placeholder for actual implementation
    return [0.0] * 10  # Example data

def get_odom_data(robot_id):
    # Placeholder for actual implementation
    return [0.0, 0.0, 0.0]  # Example data

def NeuralNetwork(laser_data, odom_data):
    # Placeholder for actual neural network processing
    return [1.0, 0.5]  # Example output data

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Access the robot_id parameter
        self.declare_parameter('robot_id')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        # Create a publisher that will publish to the cmd_vel topic of the robot
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.publish_twist)
        
        # Initialize the linear and angular velocities
        self.linear_x = 0.0  
        self.angular_z = 0.0

        # Create a subscriber for the neural directions
        self.neural_directions_sub = self.create_subscription(
            Float64MultiArray, 
            f'/Robot{self.robot_id}/neural_directions', 
            self.neural_directions_callback, 
            1
        )
        
    def publish_twist(self):
        msg_cmd = Twist()
        
        # Set the linear and angular velocities of the robot
        msg_cmd.linear.x = self.linear_x
        msg_cmd.angular.z = self.angular_z
        
        # Publish the velocities to the cmd_vel topic of the robot
        self.cmd_vel_pub.publish(msg_cmd)

    def neural_directions_callback(self, msg):
        # Assuming msg.data is a list where the first element is linear_x and the second is angular_z
        if len(msg.data) >= 2:
            self.linear_x = msg.data[0]
            self.angular_z = msg.data[1]

def run_command(num_robot, runs=0):
    rclpy.init()
    node = rclpy.create_node(f'state_{num_robot}')
    state_publisher = node.create_publisher(Float64MultiArray, f'/Robot{num_robot}/neural_directions', 10)    
    rate = node.create_rate(1)  # Publishing rate in Hz

    while rclpy.ok():
        msg = Float64MultiArray()
        
        # 1.1 Laser Scanner Data:
        laser_data = get_laser_data(num_robot)
                
        # 1.2 Odom Data:
        odom_data = get_odom_data(num_robot)
        
        msg.data = NeuralNetwork(laser_data, odom_data)
        state_publisher.publish(msg)
        rate.sleep()  # Sleep to maintain the loop rate

    node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # Start the command in a separate thread
    command_thread = threading.Thread(target=run_command, args=(1,))
    command_thread.start()
    
    # Allow time for setup
    time.sleep(5)
    
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
