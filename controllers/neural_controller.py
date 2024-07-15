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
        self.start_time = time.time()        
        self.init_flag = 0
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
        self.laser_data = []
        self.odom_data_position = [0.0, 0.0, 0.0]
        self.odom_data_orientation = [0.0, 0.0, 0.0, 0.0]
        self.odom_data_linear_velocity = [0.0, 0.0, 0.0]
        self.odom_data_angular_velocity = [0.0, 0.0, 0.0]

        # ROS Confrimation Variables (Ensuring Data is Received, Orientation is Upright, Previous Position, etc.)
        self.initial_data_received = False
        self.angle = 0.0
        self.not_upright_counter = 0
        self.upright_threshold = 5 / 0.2
        self.previous_position = None
        self.sbp = [0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 
                    1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020]
        
        self.history = [-1] * 4 * 20
        self.rewards = []
        self.episode = []
        
        # ROS Subscribers
        self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
        self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)
        
        # Confirm Intialization
        self.init_flag+=1
        self.get_logger().info(f"ROS2 Node Initialized. {self.init_flag} / 3 Initializations Complete.")

        
    def init_neural_network(self):
        # Input Size: 1 Robot_ID + ( 20 Data * 5 History Length) = 101 per Robot
        input_size = self.total_robots * 101
        
        # Neural Network Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-1, input_shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5), 
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2) 
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.load_neural_network_data()
        
        # Confirm Intialization
        self.init_flag+=1
        self.get_logger().info(f"Neural Network Initialized. {self.init_flag} / 3 Initializations Complete.")
    
    def init_swarm(self):
        # Swarm Data: Total Robots -1 (Excluding Self) * 101 Data per Robot & Swarm Publisher Creation 
        self.swarm_data = [-1] * (self.total_robots - 1) * 101
        self.swarm_pub = self.create_publisher(Float32MultiArray, f'/Robot{self.robot_id}/swarm_data', 1)
        
        # Give time for other robots to create_publisher before atempting to subscribe
        time.sleep(3)
        
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
        self.odom_data_linear_velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        self.odom_data_angular_velocity = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        
        # Confirm Initial Data Received
        self.initial_data_received = True

    def swarm_callback(self, msg):
        # Update Swarm Data
        if self.initial_data_received:
            # Determine if the robot is within distance of the robot sending the data
            pos1 = self.odom_data_position
            pos2 = msg.data[8:11]
            
            # Calculate the Euclidean Distance between the two robots
            distance = math.sqrt((pos1[0] - pos2[0])**2 + 
                                 (pos1[1] - pos2[1])**2 + 
                                 (pos1[2] - pos2[2])**2)
            
            # Update the Swarm Data if the robot is within the threshold distance (Special Case for final robot in swarm - takes position of self)
            if distance <= self.swarm_distance_threshold:
                index_start = self.sbp[self.robot_id - 1]
                index_end = self.sbp[self.robot_id]
                
                if msg.data[0] == self.total_robots:
                    self.swarm_data[index_start:index_end] = msg.data
                else:
                    index_start = self.sbp[int(msg.data[0]) - 1]
                    index_end = self.sbp[int(msg.data[0])]
                    self.swarm_data[index_start:index_end] = msg.data
            else:
                if msg.data[0] == self.total_robots:
                    index_start = self.sbp[self.robot_id - 1]
                    index_end = self.sbp[self.robot_id]
                    self.swarm_data[index_start:index_end] = [-1] * (index_end - index_start)
                else:
                    index_start = self.sbp[int(msg.data[0]) - 1]
                    index_end = self.sbp[int(msg.data[0])]
                    self.swarm_data[index_start:index_end] = [-1] * (index_end - index_start)
    
    def get_input_data(self):
        # Publish Swarm Data
        msg_swarm = Float32MultiArray()  
        msg_swarm.data = []
        msg_swarm.data.append(self.robot_id)
        msg_swarm.data.extend(self.laser_data)
        msg_swarm.data.extend(self.odom_data_position)
        msg_swarm.data.extend(self.odom_data_orientation)
        msg_swarm.data.extend(self.odom_data_linear_velocity)
        msg_swarm.data.extend(self.odom_data_angular_velocity)
        msg_swarm.data.extend(self.history)
        self.swarm_pub.publish(msg_swarm)

        # Update History Data (20 Data Points Added and Removed)
        old_history = self.history
        
        new_history = [
            *self.laser_data,
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.odom_data_linear_velocity,
            *self.odom_data_angular_velocity
        ]
        self.history = new_history + old_history[:-20]
        self.previous_position = np.array(self.odom_data_position[:2])
        
        # Return Input Data
        return [
            self.robot_id,
            *self.laser_data,
            *self.odom_data_position,
            *self.odom_data_orientation,
            *self.odom_data_linear_velocity,
            *self.odom_data_angular_velocity,
            *old_history,
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
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time < 60.0:  # 60 seconds = 1 minute
            linear_x = min(linear_x, 2.0)  # Cap linear_x at 2.0

        linear_x += random.uniform(-0.05, 0.05)
        angular_z += random.uniform(-0.05, 0.05)

        return linear_x, angular_z

    def reward_function(self):
        reward = 0.0
        current_position = np.array(self.odom_data_position[:2])
        
        if self.previous_position is not None:
            position_change = np.linalg.norm(current_position - self.previous_position)
            reward += (position_change + 1) * 10.0
        
        self.previous_position = current_position 

        rotation_matrix = utils.quaternion_to_rotation_matrix(*self.odom_data_orientation)
        up_vector = utils.get_up_vector(rotation_matrix)
        self.angle = utils.calculate_inclination_angle(up_vector)

        upright_reward = 5.0
        upside_down_penalty = -20.0

        if self.angle <= 45:
            reward += upright_reward - ((upright_reward - upside_down_penalty) / 90.0) * self.angle
        else:
            reward += upside_down_penalty + ((upright_reward - upside_down_penalty) / 90.0) * (180 - self.angle)

        if self.laser_data:
            min_distance = min(self.laser_data)
            if min_distance < 0.2:
                reward -= 10

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

        # WIP - Reorient Robot
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
        else:
            self.not_upright_counter = 0

    def reorient_robot(self):
        self.get_logger().info("Reorienting the robot...")

        # Delete the robot
        delete_client = self.create_client(DeleteEntity, '/delete_entity')
        delete_request = DeleteEntity.Request(name=f'Robot{self.robot_id}')
        delete_future = delete_client.call_async(delete_request)
        rclpy.spin_until_future_complete(self, delete_future, timeout_sec=1)

        # Spawn the robot with an empty XML description
        spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        if self.cmd_vel_pub:
                self.cmd_vel_pub.destroy()
                self.cmd_vel_pub = None
            
        try:
            xml_content =   """<?xml version="1.0" ?><!-- =================================================================================== --><!-- |    This document was autogenerated by xacro from /home/lpep/ros2ws/install/lp_neural_swarm/share/lp_neural_swarm/urdf/differential_drive_cam_lidar_fisheyes.xacro | --><!-- |    EDITING THIS FILE BY HAND IS NOT RECOMMENDED                                 | --><!-- =================================================================================== --><robot name="fourWD"><!-- Defining the colors used in this robot --><material name="Black"><color rgba="0.0 0.0 0.0 1.0"/></material><material name="Red"><color rgba="1.0 0.0 0.0 1.0"/></material><material name="White"><color rgba="1.0 1.0 1.0 1.0"/></material><material name="Yellow"><color rgba="0.0 1.0 1.0 1.0"/></material><material name="Blue"><color rgba="0.0 0.0 0.8 1.0"/></material><!-- in kg--><!-- in kg--><!-- BASE-LINK --><!--Actual body/chassis of the robot--><link name="base_link"><inertial><mass value="0.7"/><origin xyz="0 0 0"/><inertia ixx="0.001698958333333333" ixy="0" ixz="0" iyy="0.003295833333333333" iyz="0" izz="0.004049791666666666"/></inertial><visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><box size="0.22 0.145 0.09"/></geometry><material name="White"/></visual><collision><origin rpy="0 0 0 " xyz="0 0 0"/><geometry><box size="0.22 0.145 0.09"/></geometry></collision></link><gazebo reference="base_link"><material>Gazebo/White</material><turnGravityOff>false</turnGravityOff></gazebo><!-- caster wheel --><link name="caster_link"><collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry><sphere radius="0.032"/></geometry></collision><visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><sphere radius="0.032"/></geometry><material name="Blue"/></visual></link><gazebo reference="caster_link"><mu1 value="0.0"/><mu2 value="0.0"/><material>Gazebo/Blue</material><turnGravityOff>false</turnGravityOff></gazebo><joint name="caster_joint" type="fixed"><origin rpy="0 0 0" xyz="0.11 0 -0.045"/><parent link="base_link"/><child link="caster_link"/></joint><!-- RIGHT WHEEL --><link name="back_right_wheel"><visual><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><geometry><cylinder length="0.025" radius="0.032"/></geometry><material name="DarkGray"/></visual><collision><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><geometry><cylinder length="0.025" radius="0.032"/></geometry></collision><inertial><mass value="0.03"/><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><inertia ixx="9.2425e-06" ixy="0" ixz="0" iyy="9.2425e-06" iyz="0" izz="1.5360000000000002e-05"/><inertia ixx="8.7002718e-03" ixy="-4.7576583e-05" ixz="1.1160499e-04" iyy="8.6195418e-03" iyz="-3.5422299e-06" izz="1.4612727e-02"/></inertial></link><gazebo reference="back_right_wheel"><mu1 value="1.0"/><mu2 value="1.0"/><!--kp  value="10000.0" />
                                    <kd  value="1.0" />
                                    <fdir1 value="1 0 0"/--><material>Gazebo/Grey</material><turnGravityOff>false</turnGravityOff></gazebo><joint name="back_right_wheel_joint" type="continuous"><parent link="base_link"/><child link="back_right_wheel"/><origin rpy="0 0 0" xyz="-0.078 -0.08499999999999999 -0.045"/><axis rpy="0 0 0" xyz="0 1 0"/><limit effort="1000" velocity="1000"/><joint_properties damping="0.0" friction="0.0"/></joint><!-- LEFT WHEEL --><link name="back_left_wheel"><visual><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><geometry><cylinder length="0.025" radius="0.032"/></geometry><material name="DarkGray"/></visual><collision><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><geometry><cylinder length="0.025" radius="0.032"/></geometry></collision><inertial><mass value="0.03"/><origin rpy="0 1.5707963267948966 1.5707963267948966" xyz="0 0 0"/><inertia ixx="9.2425e-06" ixy="0" ixz="0" iyy="9.2425e-06" iyz="0" izz="1.5360000000000002e-05"/></inertial></link><gazebo reference="back_left_wheel"><mu1 value="1.0"/><mu2 value="1.0"/><!--kp  value="10000.0" />
                                    <kd  value="1.0" />
                                    <fdir1 value="1 0 0"/--><material>Gazebo/Grey</material><turnGravityOff>false</turnGravityOff></gazebo><joint name="back_left_wheel_joint" type="continuous"><parent link="base_link"/><child link="back_left_wheel"/><origin rpy="0 0 0" xyz="-0.078 0.08499999999999999 -0.045"/><axis rpy="0 0 0" xyz="0 1 0"/><limit effort="100" velocity="100"/><joint_properties damping="0.0" friction="0.0"/></joint><gazebo><!-- differential drive controller (for two wheels) --><plugin filename="libgazebo_ros_diff_drive.so" name="differential_drive_controller"><robotNamespace></robotNamespace><updateRate>30</updateRate><left_joint>back_left_wheel_joint</left_joint><right_joint>back_right_wheel_joint</right_joint><wheel_separation>0.35</wheel_separation><wheel_diameter>0.1</wheel_diameter><wheelAcceleration>1.8</wheelAcceleration><wheelTorque>200</wheelTorque><commandTopic>cmd_vel</commandTopic><odometryTopic>odom</odometryTopic><odometry_frame>odom</odometry_frame><robot_base_frame>base_link</robot_base_frame><!-- Odometry source, 0 for ENCODER, 1 for WORLD, defaults to WORLD --><odometrySource>world</odometrySource><publishWheelTF>false</publishWheelTF><publish_odom>true</publish_odom><publish_odom_tf>true</publish_odom_tf><publish_wheel_tf>true</publish_wheel_tf><publishTF>false</publishTF><publishWheelJointState>false</publishWheelJointState><legacyMode>false</legacyMode><robotBaseFrame>base_footprint</robotBaseFrame><rosDebugLevel>na</rosDebugLevel></plugin></gazebo><!--fb : front, back ; lr: left, right --><!-- ydlidar laser sensor --><link name="ydlidar_link"><collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry><cylinder length="0.055" radius="0.033"/></geometry></collision><visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><cylinder length="0.055" radius="0.033"/></geometry><material name="Blue"/></visual></link><gazebo reference="ydlidar_link"><material>Gazebo/Blue</material><turnGravityOff>false</turnGravityOff><sensor name="head_ydlidar_sensor" type="ray"><pose>0 0 0.0175 0 0</pose><visualize>False</visualize><update_rate>8.3</update_rate><ray><scan><horizontal><samples>7</samples><resolution>1</resolution><min_angle>-0.3</min_angle><max_angle>0.3</max_angle></horizontal></scan><range><min>0.10</min><max>2.0</max><resolution>0.01</resolution></range><noise><type>gaussian</type><mean>0.0</mean><stddev>0.01</stddev></noise></ray><plugin filename="libgazebo_ros_ray_sensor.so" name="scan"><topicName>scan</topicName><frameName>ydlidar_link</frameName><output_type>sensor_msgs/LaserScan</output_type></plugin></sensor></gazebo><joint name="ydlidar_joint" type="fixed"><origin rpy="0 0 0" xyz="0 0 0.0725"/><parent link="base_link"/><child link="ydlidar_link"/></joint></robot>
                            """

            initial_pose = Pose()
            initial_pose.position.x = self.odom_data_position[0]
            initial_pose.position.y = self.odom_data_position[1]
            initial_pose.position.z = 0.0
            initial_pose.orientation.x = 0.0
            initial_pose.orientation.y = 0.0
            initial_pose.orientation.z = self.odom_data_orientation[2]
            initial_pose.orientation.w = self.odom_data_orientation[3]

            spawn_request = SpawnEntity.Request(
                name=f'Robot{self.robot_id}',
                xml=xml_content,
                robot_namespace=f'/Robot{self.robot_id}',
                initial_pose=initial_pose,
                reference_frame='world'
            )

            spawn_future = spawn_client.call_async(spawn_request)
            rclpy.spin_until_future_complete(self, spawn_future, timeout_sec=1)
                        
            # TODO Reconnect publishers and subscribers
            time.sleep(1)
            self.destroy_publisher(self.cmd_vel_pub)
            self.destroy_publisher(self.swarm_pub)
            
            self.destroy_subscription(self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1))
            self.destroy_subscription(self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1))

            self.cmd_vel_pub = self.create_publisher(Twist, f'/Robot{self.robot_id}/cmd_vel', 1)
            self.swarm_pub = self.create_publisher(Float32MultiArray, f'/Robot{self.robot_id}/swarm_data', 1)

            self.create_subscription(LaserScan, f'/Robot{self.robot_id}/scan/out', self.laser_callback, 1)
            self.create_subscription(Odometry, f'/Robot{self.robot_id}/odom', self.odom_callback, 1)
            
            self.get_logger().info("Robot reoriented successfully.")
            self.not_upright_counter = 0
                
        except Exception as e:
            self.get_logger().error(f'Exception during reorientation: {e}')

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.train_neural_network()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
