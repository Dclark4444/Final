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

# Implemented Movement classes 
from defendagentattack.rl_agent_nodes import Attack_move, Defend_move 



class ExecuteOptimal(Node):
    def __init__(self, player):
        super().__init__('execute_optimal_policy', player)
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

        # intializing time and state variables
        self.action_MIN_RUN_TIME = 3.5 
        self.start_time = None
        self.step = "Accessing"


         # Fetch Actions and states (saved files)
         # Joint action --> [Attack, Defender]
         # Attack Agent can Face/Turn to face opponent (0). Move_R/L (1,2), Arm home (3) Trick Move?Attack (4,5)
         # Defend Agent can turn to face (0, Turn R/L (1,2, arm_up (3) arm_right/left (4,5)
        action_pth = os.path.join(self.share, 'matrices', 'action.txt')
        self.actions = np.loadtxt(action_pth)


        # Atack agent states are indicies 0-1 and Defend are 2-3 
        # index 0 represents the Attack Agent stance relative to Defense
        # ==> 0 - in front, 1 - to the Right, 2 - to the left 
        # index 1 represens the A Agent with Arm in Disguise (1), or Attacking (2), or just home (0)
        
        # index 2 represents the Defense Agent stance relative to Attack
        # ===> forward facing (0), to_right (1), to_left (2)
        # index 3 represents the D Agent Arm position 
        # ==> home/up (0), arm_right (1), arm_left (2)
        state_pth = os.path.join(self.share, 'matrices', 'states.txt')
        self.states = np.loadtxt(state_pth)


        # matric of states on the columns and the rows. 
        # cells reutrn action indiceies if accessible, -1 if not.
        state_nstate_pth = os.path.join(self.share, 'matrices', 'state_nstate.txt')
        self.state_nstate = np.loadtxt(state_nstate_pth)


        # Saved RL matrix --- state-action-agent matrix 
        matrix_path = os.path.join(self.share, 'matrices', 'RL_matrix.txt')
        self.q_matrix = np.loadtxt(matrix_path) # np array of q_matrix 
        self.get_logger().info(f'Loaded RL_matrix')

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


        # # subscriptions 
        # self.image_sub = self.create_subscription(CompressedImage, 
        #     compress_image_topic, 
        #     self.image_callback, #trigger image for robot to look around (movement)
        #     10
        # )

        # self.scan_sub = self.create_subscription(LaserScan, scan_topic, 
        #     self.scan_callback,10)
        

        timer_period =  0.5
        # generating policy loop, can adjust how often function is being called
        self.timer = self.create_timer(timer_period, self.policy_loop)
        
        # Player and policy execution variables
        self.players = ["Attack", "Defend"]  # None only when it hasnt chosen a player yet
        self.player_idx = player # getting player index 
        self.joint_action_idx = None  # update 
        self.curr_state_idx = 0 # update 

        self.fourht_wall_path =  os.path.join('.', 'fourth_wall.txt') # text file to communicate
        self.fourth_wall_file = np.loadtxt(self.fourht_wall_path)
        self.reading_file() # readint he file to get the actions 
        # in the mean time also performing policy loops 

        # both robots should know who is who 
        time.sleep(0.5)



    def get_action(self): 
        curr_state = self.curr_state_idx # get the current state action 
        # agent = self.players[self.player_idx] # get the agent id 
        agent_qs = self.RL_matrix[curr_state, :, self.player_idx]
        ## FOR ALL THE VALID ACTIONS of the agent 
        action_idx = int(np.argmax(agent_qs))
        return action_idx 


    def reading_file(self):

        curr_state = self.curr_state_idx # get the current state index 
        player = self.players[self.player_idx] 

        if self.joint_action_idx != None:  # only read file if i have t gotten a joint action_IDX 
            return 

        filesize = os.path.getsize(self.fourth_wall_path)
        if filesize == 1: 
            pos_action = np.loadtxt(self.fourth_wall_path)

        if filesize > 1:

    #   can find the joint action from finding the joint action in the joint_action list, and pass that shit 
            self.joint_action_idx = np.where(self.actions == pos_action) # get the joint action index 

    # detete the joint actions once saved 
            np.savetxt(self.fourth_wall_path, np.array([])) 
        
        else: 
            pos_action = np.array([])
        
        action_idx = None 

        if player == "Attack":
            if np.isnan(pos_action[self.player_idx]):
                self.get_action()
        else: 
            if not np.isnan(pos_action[0]):
                self.get_action()
        
        pos_action[self.player_idx] = action_idx  # update the next possible action that could be performed 
        np.savetxt(self.fourth_wall_path, pos_action, fmt="%d") # need to clear this every time an action is performed 

    

    def policy_loop(self):
        player = self.players[self.player_idx]

        if self.joint_action_idx is None: # keep reading the file 
            self.reading_file()
            # perform the action 

        else: 
            curr_at_player   = self.actions[self.joint_action_idx][self.player_idx]
            # RUN THE MOVEMENT NODE STUFF HERE ---> JUST RUN ACTION(action), can be initilized with the action 

            
            if player == "Attack": # run the movement upon initilization ---> hopefully continues until its told to stop 
                self.move_action = Attack_move(curr_at_player)
            # if ther eis somethign in there, get the action for your robot and move it 
            else: 
                self.move_action = Defend_move(curr_at_player)


        # once it performs the action complelely, then highlight the new state, using hte action_state

        pos_n_state = self.state_nstate[self.curr_state_idx] # given the states and nex state comparisons 
        # which state contians the action you performed 
        next_state_idc = np.where(pos_n_state  == self.actions[self.joint_action_idx]) #

        self.curr_state_idx = next_state_idc
        self.joint_action_idx = None 



def main(args=None):
    rclpy.init(args=args)
    TYPE = input("Press 1 to load defender, press 0 to load attacker ") ## input the player 
    if TYPE == '1':
        TYPE = 'Defense'
    else:
        TYPE = 'Attack'


    FILE = "fourth_wall.txt"
    lst_msg = "" # just starting to write in for both robots 
    if os.path.exists(FILE): 
        with open(FILE, "r") as f:
            msg =f.read().strip() # get_message                 

        if msg != lst_msg: # if something has been written in and we can see it 
     # if message is not none, comfirming you are the secoind robot reading the file 
            print(f'{TYPE} has read and has connection with other robot')
            node = ExecuteOptimal(TYPE) # set node after both robots  have read eachother. 
            
            # after acknowledgement of connection ----> delete content for fighting 
            with open(FILE, "w") as f: 
                f.write('') # empty 


        else: 
            # make a msg and wait for the other TYPE to respond back 
            with open(FILE, "w") as f:
                f.write(f'{TYPE} connected')
            print(f'{TYPE} connected')
    

    else:
        print('its not working g')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


