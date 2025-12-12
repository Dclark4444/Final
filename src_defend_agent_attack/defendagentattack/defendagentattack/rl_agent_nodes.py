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



class MovementNode_RL(Node, ABC):
    """ Superclass containing all shared attributes, initializations, and helper methods. """
    
    def __init__(self, node_name, action):
        # initialize with the current action 
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


        # Shared Timing/Action Variables
        self.action_MIN_RUN_TIME = 2.0  # seconds 
        self.start_time = None  
        
        # shared varible names 
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

        #Topics 
        cmd_vel_topic = f"/tb{ros_domain_id}/cmd_vel"
        # compress_image_topic = f'/tb{self.ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        scan_topic = f'/tb{self.ros_domain_id}/scan'
        arm_angles = f'/tb{ros_domain_id}/target_joint_angles'

        # publishing
        self.joint_arm_pub = self.create_publisher(ArmJointAngles, arm_angles, 10)
        self.movement_publisher =  self.create_publisher(Twist, cmd_vel_topic, 10)

        # subscribing to 
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, 
            self.scan_callback,10)


    def body_movement(self, neg_velo, forward=False): 
        """Helper function for body movement of any agent,inludes optional conditionals
          as parameters, allowing the generalization of body movement"""
        #if velocity is negative then it will turn left 
        x = -1 if neg_velo else 1
        twist_msg = Twist()
        twist_msg.angular.z = x*self.angvelocity# tbh i don't know which way is left or not 
        
        if forward: 
            twist_msg.linear.x = self.linear_v
        self.movement_publisher.publish(twist_msg)


    def arm_move(self,left_joint_1, pose):
        """Helper function for arm movement of any agent"""
        # if joint angle negative ==== Guarding left 
        x = -1 if left_joint_1 else 1  
        # left_joint_1 == True just negate the first joint (reflecting it across axis of rotation 
        arm_msg = ArmJointAngles()
        arm_msg.joint1 = x*pose[0] 
        arm_msg.joint2, arm_msg.joint3, arm_msg.joint4 = pose[1:3]
        self.joint_arm_pub.publish(arm_msg)


    def scan_callback(self, scan:LaserScan):
        """"Triggered everytime robot receives scan. Function controls for the length of time 
        to perform the action, and assesses whether agents are at unsafe distance wiht scan.ranges """
        
        self.get_logger().debug('LaserScan received')
        scan_ranges = np.array(scan.ranges)
        num_points = len(scan_ranges)

        # Define slices for forward direction
        window = max(5, num_points // 25) 

        # forward facing indices 
        forward_slice = np.concatenate((scan_ranges[:window],scan_ranges[-window:]))

        valid = forward_slice[np.isfinite(forward_slice)]
        if len(valid) == 0:
            self.get_logger().warn("No valid scan points in front.")
            return

        # how far is the opponent when its in front of you 
        self.scan_dist = float(np.median(valid)) 

        if self.start_time != None: 
            # start time (first scan, after the action has been called 
            self.start_time = self.get_clock().now()

        # how much time has passed from intiial call of action and present call 
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

#       acknowledges that the action SHOUOLD be done 
        if elapsed > self.action_MIN_RUN_TIME:
            self.get_logger().info("Action should have finished performing")
            
            if (self.scan_dist - self.desired_scan_dist) < 0.5:
                # if the robots are too close to eachother after action is performed, shutdown node
                rclpy.shutdown()

            # resest the self.start_time 
            self.start_time = None 

        # set scan as a global variable to potentially use elsewhere
        self.scan = scan



    def finding_face(self):
        """Finds the angle difference between the face of the associated agent, and the face of opponent"""

        ranges = np.array(self.scan.ranges)
    
    # filter out inf values 

        min, max = (0.5, 1.0)

    # what indices do we see valid ranges (given min and max values)
        valid_indices = np.where((ranges > min) & (ranges < max) & np.isfinite(ranges))[0]
    
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
    """Child class: specific for Attack Movements given an action"""
    def __init__(self, action):
        super().__init__('attack_move_nn', action)

        self.action_idx_to_name = ['home', 'M_L', 'M_R', 'arm_home', 'A_D', 'A_A']


        self.cur_action = self.action_idx_to_name[action] # agent-specific action 
    

        self.A_attack_arm_pose = [0.530, 0.32, 0.150, -0.25] # potentially fix these

        self.desired_angle = 60  # desired body turn angle 
        self.angvelocity = self.desired_angle/(self.action_MIN_RUN_TIME-0.5) 
        # velocity bsed on desired anble and time limit

        self.desired_dist = 0.3 # meters 
        self.linear_v = self.desired_dist/self.action_MIN_RUN_TIME
        # THis is arm pose for R but Left is negative of the first joint 


        self.timer = self.create_timer(self.timer_period, self.policy_loop)
        # call self.policy_loop every 0.5 seconds 

        self.run_action() # run the action 


    
    def run_action(self):

        if 'M' == self.curr_at[0]:  # if the action is in the body 
            if 'L' in self.curr_at: # if left, neg_velo is true, negating the first joint 
                self.body_movement(neg_velo=True)
            else: 
                self.body_movement(neg_velo=False)
            

        if 'A' ==  self.curr_at[0]: # if the action is in the arm 
            
            if 'D' in self.curr_at: # do trick -- opposite arm pose than attack 
                self.arm_move(True, self.G_R_arm_pose) 
           
            else:  # If attacking 
                self.arm_move(False, self.G_R_arm_pose)
            
            
        elif self.curr_at == 'home': # reversed 
           
            # facing the opponnet is going home 
            angle_error = self.fnding_face()

            msg = Twist()
            msg.angular.z = angle_error/(self.action_MIN_RUN_TIME - 0.5) 
            self.movement_publisher.publish(msg)

        else: # if action is arm_home, move arm to home pose 
            self.arm_move(False, self.home_pose)
    


class Defense_move(MovementNode_RL):
    """Child class: specific for Attack Movements given an action"""

    def __init__(self, action):
        super().__init__('defense_move_nn', action)

        self.action_idx_to_name = ['home', 'M_L', 'M_R', 'home_arm', 'A_R', 'A_L']
        self.cur_action = self.action_idx_to_name[action] # get player-specific action 
                
        # self.curr_at = action # set action_idx value to curr_at

        self.G_R_arm_pose = [1.33, 0.4, 0.2, 0.8] # move arm to the right 
        self.desired_angle = 30 # angle of turn 
        self.angvelocity = self.desired_angle/(self.action_MIN_RUN_TIME - 0.5) # set angular velocy 
        # based on minimum run time and desired angle 

        timer_period =  0.5
        self.timer = self.create_timer(timer_period, self.policy_loop)
        # create loop that is called every 0.5 seconds to access 
        
        self.run_action()



    def run_action(self):
        """Similar to run_action of Attack_move"""

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