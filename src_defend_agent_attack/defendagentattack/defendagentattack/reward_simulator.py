import os
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory 
from rclpy.node import Node

from AttackDefend_interfaces.msg import Reward
from AttackDefend_interfaces.msg import Action 



## WRITING TO FILE AND READING FILE TO OBTAIN ACTION AND THEN ReTURN 

class LearningMatrix(Node):
    def __init__(self):
        super().__init__('simulator for RL')

        self.share = get_package_share_directory('q_learning_project')

        actions_path = os.path.join(self.share, 'matrices', 'actions.txt')
        self.actions = np.loadtxt(actions_path)

        # q matrix states
        states_path = os.path.join(self.share, 'matrices', 'states.txt')
        self.states = np.loadtxt(states_path) # np array of all the sttes 
       
        #different types of rewards
        self.attack_reward = 50  # large reward for popping the balloon (for attacker)
        self.attack_damn = -5 # small conseequence for attack being blocked 
        self.defend_reward = 15 # small reward for blocking hit 
        self.defend_damn = -self.attack_reward # large consequence for getting attacked 
        self.default = 0 
        self.safe = 5  # going home is safe 


        # goal states for attacker / defender: 
        # goal of attacker is negative reward of defender 
        self.goal_attack = [[2,2,0,0], [1,2,0,0]] # when it has attacked when robot is no turned to it or shield is not in the way 
        self.goal_defend = [[2,2,2,0],[2,2,0,2], [1,2,0,1], [1,2,1,0]] #  when it has successfully blocked the attack  --- > turned int eh same direction and arm in direction of attack 

        self.state_to_act = {0: [0,1,2], 1: [0,3,4]}

        # keep track of the iteration number
        self.iteration_num = 0


        # initialized state 
        self.curr_state = self.states[0]     #gett

        # get the ROS_DOMAIN_ID aka robot number and format as two digits
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID, {ros_domain_id}')

        # set up a subscription to the action topic that will trigger reward sending
        action_topic = f"/tb{ros_domain_id}/multiagent/action_id"
        self.create_subscription(
            Action,
            action_topic,
            self.send_reward,
            10,
        ) 
        # subscribe to actions from multi-agent, 
        # should recieve actions of both attackre and defender 

        # set up the reward publisher
        reward_topic = f"/tb{ros_domain_id}/multiagent/reward"
        self.reward_pub = self.create_publisher(Reward, reward_topic, 10)


    def helper_val(self, curr_player_st, act_for_play):
        """Maps actions to states"""
        
        if act_for_play > 2:  
            idx_for_st = 1
        else:
            idx_for_st = 0 
        
        find_act = [i for i, state in enumerate(self.state_to_act[idx_for_st]) if state.tolist() == self.state_to_act[idx_for_st]][0]
        # get value for the index that changes 

        curr_player_st[idx_for_st] = find_act

        return curr_player_st


    def send_reward(self, msg:Action): 
# reward function for Reinforcement Learning 
        # state_to_act = {0: [0,1,2], 1: [0,3,4]}

        r_A  = self.default 
        r_D = self.default


        # load message (logger)
        self.get_logger().info(f'Recieved action: robot_object = {msg.action_id}')


        # Determine the next joint state based on action (received action)
        A_cur_st = self.curr_state[0:2]
        D_cur_st = self.curr_state[2:4]


        a_act = msg.action_id[0] # ok if the action here is  3 --> 
        d_act = msg.action_id[1]

        a_state = self.helper_val(A_cur_st, a_act)
        d_state = self.helper_val(D_cur_st, d_act)

        next_st = a_state + d_state

        # attacker goal checl
        home_st = [0,0,0,0]
        if self.curr_state != home_st and next_st == home_st: # tiny reward for resetting to home 
            r_A += self.safe
            r_D += self.safe 
        if next_st in self.goal_attack:
            r_A += self.attack_reward
            r_D += self.defend_damn

        elif next_st in self.goal_defend: # defender goal check 
            r_A += self.attack_damn
            r_D += self.defend_reward

        # combining to make a joint reward (list)
        joint_reward = [r_A, r_D]


        # update == =and publish
        reward_msg = Reward 
        reward_msg.reward = joint_reward
        self.reward_pub.publish(reward_msg)
        self.curr_state = next_st


def main(args=None):
    rclpy.init(args=args)

    node = LearningMatrix()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

        