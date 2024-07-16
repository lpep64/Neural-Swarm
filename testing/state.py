#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class LidarState(Node):
    def __init__(self):
        super().__init__('lidar_state')
        self.robot_num = self.declare_parameter('robot_num', 1).value
        self.get_logger().info('Running robot number: %d' % self.robot_num)        
        

def main(args=None):
    rclpy.init(args=args)
    node = LidarState()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error('Unhandled exception: %s' % (e,))
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()