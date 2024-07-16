#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self, robot_id):
        super().__init__(f'robot_controller{robot_id}')
        
        # Create a publisher that will publish to the cmd_vel topic of the robot
        self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{robot_id}/cmd_vel', 1)
        
        # Create a timer that calls the publish_twist method every 0.1 seconds
        self.timer = self.create_timer(0.1, self.publish_twist)

    def publish_twist(self):
        msg_cmd = Twist()
        
        # Set the linear and angular velocities of the robot
        msg_cmd.linear.x = 1.0
        msg_cmd.linear.y = 0.0
        msg_cmd.linear.z = 0.0
        msg_cmd.angular.x = 0.0
        msg_cmd.angular.y = 0.0
        msg_cmd.angular.z = 1.0
        
        # Publish the velocities to the cmd_vel topic of the robot
        self.cmd_vel_pub.publish(msg_cmd)

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    
    # Runs code over all 20 robots rather than just one
    for i in range(1, 21):
        # Create a RobotController for each robot and add it to the executor
        robot_controller = RobotController(i)
        executor.add_node(robot_controller)
        
    try:
        # Start the executor to begin processing ROS callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()