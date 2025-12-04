import rclpy
import sys
from rclpy.node import Node
import time

import cv2
import cv_bridge
import numpy as np
import os

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from omx_cpp_interface.msg import ArmGripperPosition, ArmJointAngles


class ExecuteOptimal(Node):
    def __init__(self, player):
        super().__init__('execute_optimal_policy')
        # getting shared directory to be able to access q_matrix


         # get ROS_DOMAIN_ID
        ros_domain_id = os.getenv("ROS_DOMAIN_ID")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"

        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')   

        # Player and policy execution variables
        self.players = ["Attack", "Defend"]  # None only when it hasnt chosen a player yet
        self.player_idx = None
        self.player_exp_actions = {"attack": ["home","M_R","M_L","arm"],
                              "defend" : ["home", "A_R", "A_L"]}
        
        self.start_time = None

    
        self.step = "Accessing"
        
        self.curr_at = None
        self.past_at = None 


         # Fetch Actions and states (saved files)
         # Joint action --> [Attack, Defender]
         # Attack Agent can Face/Turn to face opponent (0). Move_R/L (1,2), arm down in direction of move (3)
         # Defend Agent can Turn to Face (0), arm_back_right/left (1,2) 
        action_pth = os.path.join(self.share, 'matrices', 'action.txt')
        self.actions = np.loadtxt(action_pth)


        # Atack agent states are indicies 0-1 and Defend are 2-3 
        # index 0 represents the Attack Agent stance relative to Defense
        # ==> 0 - in front, 1 - to the Right, 2 - to the left 
        # index 1 represens the A Agent arm up (0), or down (1)
        # index 2 represents the D Agent stance relative to Defense
        # ===> forward facing (0), not seeing (1)
        # index 3 represents the D Agent Arm position 
        # ==> home/up (0), arm_down_right (1), arm_down_left (2)
        state_pth = os.path.join(self.share, 'matrices', 'states.txt')
        self.states = np.loadtxt(state_pth)



        matrix_path = os.path.join(self.share, 'matrices', 'RL_matrix.txt')
        self.q_matrix = np.loadtxt(matrix_path) # np array of q_matrix 
        self.get_logger().info(f'Loaded RL_matrix')


        # color bounds of all items in the environment 
        self.color_bounds = {
        "spear":  (np.array([140, 50, 50]), np.array([170, 255, 255])),
        "shield": (np.array([35, 50, 50]),  np.array([85, 255, 255])),
        "balloon":  (np.array([85, 50, 50]), np.array([125, 255, 255]))
        }

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()


        #Topics 
        cmd_vel_topic = f"/tb{ros_domain_id}/cmd_vel"
        compress_image_topic = f'/tb{self.ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        scan_topic = f'/tb{self.ros_domain_id}/scan'
        arm_grip = f'/tb{ros_domain_id}/target_gripper_position'
        arm_angles = f'/tb{ros_domain_id}/target_joint_angles'


        # publishing
        self.joint_arm_pub = self.create_publisher(ArmJointAngles, arm_angles, 10)
        self.gripper_pub = self.create_publisher(ArmGripperPosition, arm_grip, 10)
        self.movement_publisher =  self.create_publisher(Twist, cmd_vel_topic, 10)


        # subscriptions 
        self.image_sub = self.create_subscription(CompressedImage, 
            compress_image_topic, 
            self.image_callback, #trigger image for robot to look around (movement)
            10
        )

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, 
            self.scan_callback,10)
        


        timer_period =  0.5
        self.timer = self.create_timer(timer_period, self.policy_loop)
        
        
        
        time.sleep(2)



        self.send_join_info()

        # Wait until 2 robots are running this file
        # Read file and confirm who's who--- perform an action 

        self.perform_action()


    def perform_action(self, action): 
        # Execution Lock of action until new state is achieved ==> hypothesized to take 3 seconds 
 
        MIN_RUN_TIME = 3.5 

        if self.start_time != None: 
            self.start_time = self.get_clock().now()
        self.move_robot( action)

            
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

     # first instance of 3.5 second run, go to the next action, after accesing state
        if elapsed > MIN_RUN_TIME:
            self.get_logger().info("Action have finished performing")
            self.step == "Accessing"
            self.start_time = None 


            
    def move_robot(self, action):
        if self.player_idx == 0:  # Attack 
            # how to do their action 
            r_act = action[self.player_idx] # get the specific action 

            lat_velocity = 0.25 # R = +, L = - 
            good_dist =

            if r_act == "home":
                # making sure you are facing the other person 
                None
            
            if r_act[0] == "M":
                # moving to teh right or left 
                if r_act[2] == "R":
                    None 
            
            else: # arm going down 
                None 
    

            # get the name of the action 
        if self.player == 1: # Defend action 
            r_act = action[self.player_idx]
            
            if r_act == "home": # Making sure its facing the other person
                None
            

            if r_act[2] == "R": # Turning Right 
                None 

            else:  # Turning Left
                None 




    def send_join_info():
        # if nothing was written then you will be the attack player
        return 
    

    def policy_loop(self): # RUns every once in a while, concurrent with scan  and image_callback
        if self.step == "Moving":      # currently performing action  
            self.perform_action()



    def image_callback(self, msg: CompressedImage):
        """ Process each incoming image and update object centroid continuously.
        Works while looking for the object or while moving toward it.
        """


        if self.state == "Accessing":
            player = self.players[self.player_idx]


            # get the player ---> make steps to acess the state of the other robot

        else: 
            None 



    def scan_callback(self, scan:LaserScan):
        """ Triggered when subscribed to LaserScan, only when going to the desired location, 
        When close enough the location, call the function using_arm. 
        """
        # self.get_logger().info(f'LaserScanning in scan_callback, and this is state {self.state}')
        
        # need to use it when assessing state 
        if self.state == "Moving":
            # when moving i need to make sure im not bumping into seomething 
            None 
        
        # when moing, I probably don't need scan ranges? 
        # --> I think i can account for lateral distance movement
       
       
        if self.state == "Accessing": # distinct things looking for based on the states.
            None 
            #Front facing, TO teh right, to the Left? ===> Uses other robot to define states of another 
        # class variables are dictionarities accounting for both agents


        # update class variables here




def main(args=None):
    rclpy.init(args=args)
    # player = sys.argv[1:] # Attack (A), Defend (D)
    # node = ExecuteOptimal(player)
    node = ExecuteOptimal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()





