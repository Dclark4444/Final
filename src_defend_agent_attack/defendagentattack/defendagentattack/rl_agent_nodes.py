import rclpy
import sys
from rclpy.node import Node
from abc import ABC, abstractmethod
import time
import os

import cv2
import cv_bridge
import numpy as np

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from omx_cpp_interface.msg import ArmGripperPosition, ArmJointAngles



# Actions for RL Learning, A couple tweaks from the NN movememtns

# Defense turn 60 degrees R or Left, turn 45 degrees R or L 

# Attack M_R/M_L: move 0.75 meter sand turn 45 degrees, 
           # Arm Distract: arm position opposite of attack, aArm attack: move arm donw  (A_A, A_D)

# HOme: Arm up, rever last action 



class MovementNode_RL(Node, ABC):
    """ Superclass containing all shared attributes, initializations, and helper methods. """
    
    def __init__(self, node_name, action):
        super().__init__(node_name, action)
        # 3. Get ROS_DOMAIN_ID (Shared setup)
        ros_domain_id = os.getenv("ROS_DOMAIN_ID")
        try:
            domain_id_int = int(ros_domain_id)
            if domain_id_int < 10:
                self.ros_domain_id = "0" + str(domain_id_int)
            else:
                self.ros_domain_id = str(domain_id_int)
        except Exception:
            self.ros_domain_id = "00"

        self.get_logger().info(f'ROS_DOMAIN_ID: {self.ros_domain_id}')  


        # 4. Shared Timing/Action Variables
        self.action_MIN_RUN_TIME = 2.0  # seconds 
        self.start_time = None 
        
        # 5. Shared home pose 
        self.home_pose = [0.0, 0.0, 0.0, 0.0] # FILL THIS IN 
        self.scan_dist = None 
        self.desired_scan_dist = 0.5 # Desired dist is 0.5 meters away 
        
        # diff variables for Attack and Defend 
        self.G_R_arm_pose = [0,0,0,0] # NEED TO FILL IN 
        self.desired_angle = 0.0 
        self.angvelocity = 0.0 
        self.desired_dist = 0.0
        self.linear_v = 0.0

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # actions are 3 for defense: 
        #Guard Left, Guard Right and Reverse 

        #Topics 
        cmd_vel_topic = f"/tb{ros_domain_id}/cmd_vel"
        # compress_image_topic = f'/tb{self.ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        scan_topic = f'/tb{self.ros_domain_id}/scan'
        arm_grip = f'/tb{ros_domain_id}/target_gripper_position'
        arm_angles = f'/tb{ros_domain_id}/target_joint_angles'


        # publishing
        self.joint_arm_pub = self.create_publisher(ArmJointAngles, arm_angles, 10)
        self.gripper_pub = self.create_publisher(ArmGripperPosition, arm_grip, 10)
        self.movement_publisher =  self.create_publisher(Twist, cmd_vel_topic, 10)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, 
            self.scan_callback,10)


    def body_movement(self, neg_velo, forward=False): # if velocity is negative then it will turn left 
        x = -1 if neg_velo else 1
        twist_msg = Twist()
        twist_msg.angular.z = x*self.angvelocity# tbh i don't know which way is left or not 
        
        if forward: 
            twist_msg.linear.x = self.linear_v
        self.movement_publisher.publish(twist_msg)


    def arm_move(self,left_joint_1, pose):
        # if joint angle negative ==== Guarding left 
        x = -1 if left_joint_1 else 1 
        arm_msg = ArmJointAngles()
        arm_msg.joint1 = x*pose[0] 
        arm_msg.joint2, arm_msg.joint3, arm_msg.joint4 = pose[1:3]
        self.joint_arm_pub.publish(arm_msg)


    def scan_callback(self, scan:LaserScan):
        self.get_logger().debug('LaserScan received')
        scan_ranges = np.array(scan.ranges)
        num_points = len(scan_ranges)

        # Define slices for forward direction
        window = max(5, num_points // 25) 

        forward_slice = np.concatenate((scan_ranges[:window],scan_ranges[-window:]))

        valid = forward_slice[np.isfinite(forward_slice)]
        if len(valid) == 0:
            self.get_logger().warn("No valid scan points in front.")
            return

        self.scan_dist = float(np.median(valid)) 

        if self.start_time != None: 
            self.start_time = self.get_clock().now()
            
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

     # first instance of 3.5 second run, go to the next action, after accesing state
        if elapsed > self.action_MIN_RUN_TIME:
            self.get_logger().info("Action should have finished performing")
            
            if (self.scan_dist - self.desired_scan_dist) < 0.5:
                rclpy.shutdown() ## immediate shutdown if the robots are very close to eachother

            self.start_time = None 

            # rclpy.shutdown() automatic shutdown of node once it hits a time limit, so execution of policy will occur

        self.scan = scan



    def finding_face(self):

        ranges = np.array(self.scan.ranges)
    
    # filter out inf values 
        valid_indices = np.where((ranges > self.scan.range_min) & (ranges < self.scan.range_max) & np.isfinite(ranges))[0]
    
        if len(valid_indices) == 0:
            return None
        
    # find the index corresponding to the minimum distance
        min_range_index = valid_indices[np.argmin(ranges[valid_indices])]

        # angle error 
        return self.scan.angle_min + (min_range_index * self.scan.angle_increment)
        

    def policy_loop(self):
        self.run_action()

    @abstractmethod
    def run_action(self):
        raise NotImplementedError("run_action method must be implemented by the child class (Defense/Attack).")
        

    

class Attack_move(MovementNode_RL):
    def __init__(self, action):
        super().__init__('attack_move_nn', action)

        self.action_idx_to_name = ['home', 'M_L', 'M_R', 'arm_home', 'A_D', 'A_A']


        self.cur_action = self.action_idx_to_name[action]

        # tie variables for performing actions (limit)
        self.action_MIN_RUN_TIME = 2 # seconds 
        self.start_time = None
    
        # self.step = "Accessing"

        self.A_attack_arm_pose = [0,0,0,0] # GIve this values 

        self.desired_angle = 60 
        self.angvelocity = self.desired_angle/(self.action_MIN_RUN_TIME)

        self.desired_dist = 0.3 # meters 
        self.linear_v = self.desired_dist/self.action_MIN_RUN_TIME
        # THis is arm pose for R but Left is negative of the first joint 


        timer_period =  0.5
        self.timer = self.create_timer(timer_period, self.policy_loop)
    
        time.sleep(1)
        self.run_action()


    
    def run_action(self):

        if 'M' == self.curr_at[0]:  # if the action is in the body 
            if 'L' in self.curr_at: 
                self.body_movement(neg_velo=True)
            else: 
                self.body_movement(neg_velo=False)
            

        if 'A' ==  self.curr_at[0]: # if the action is in the arm 
            
            if 'D' in self.curr_at: # If distracting 
                self.arm_move(True, self.G_R_arm_pose)
           
            else:  # If attacking 
                self.arm_move(False, self.G_R_arm_pose)
            
            
        elif self.curr_at == 'home': # reversed 
            # facing the opponnet is going hoem 
            angle_error = self.fnding_face()
            msg = Twist()
            msg.angular.z = angle_error/(self.action_MIN_RUN_TIME - 0.25) 
            self.movement_publisher.publish(msg)

        else: # if action is arm_home 
            self.arm_move(False, self.home_pose)
    


class Defense_move(MovementNode_RL):
    def __init__(self, action):
        super().__init__('defense_move_nn', action)

        self.action_idx_to_name = ['home', 'M_L', 'M_R', 'home_arm', 'A_R', 'A_L']
        self.cur_action = self.action_idx_to_name[action]

        
        self.action_MIN_RUN_TIME = 2 # seconds 
        self.start_time = None
            
        self.curr_at = action

        self.G_R_arm_pose = [1.33, 0.4, 0.2, 0.8]
        self.desired_angle = 30 
        self.angvelocity = self.desired_angle/(self.action_MIN_RUN_TIME)

        timer_period =  0.5
        self.timer = self.create_timer(timer_period, self.policy_loop)
        

        time.sleep(2)
        self.run_action()



    def run_action(self):

        if 'M' == self.curr_at[0]:  # if the action is in the body 
            if 'L' in self.curr_at: 
                self.body_movement(neg_velo=True)
            else: 
                self.body_movement(neg_velo=False)
            

        if 'A' ==  self.curr_at[0]: # if the action is in the arm 
            
            if 'L' in self.curr_at:
                self.arm_move(True, self.G_R_arm_pose)
            else: 
                self.arm_move(False, self.G_R_arm_pose)
            
            
        elif self.curr_at == 'home': # reversed 
            # facing the opponnet is going hoem 
            angle_error = self.fnding_face()
            msg = Twist()
            msg.angular.z = angle_error/(self.action_MIN_RUN_TIME - 0.25) 
            self.movement_publisher.publish(msg)

        else: # if action is arm_home 
            self.arm_move(False, self.home_pose)



def main(args=None):
    rclpy.init(args=args)
    # player = sys.argv[1:] # Attack (A), Defend (D)
    node = MovementNode_RL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()