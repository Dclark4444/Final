import os
import rclpy
import sys
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import numpy as np # type: ignore
import random

from AttackDefend_interfaces.msg import Reward, Action 
from rclpy.qos import qos_profile_sensor_data
import time 

def joint_action_to_index(a_A, a_D):
    """Encodes the (a_A, a_D) pair into a single index (0 to 11)."""
    return a_A + 5 * a_D # both attacker and defending have 4 acions


class ReinforceLearn(Node):
    def __init__(self):
        super().__init__('reinforcement learning')
        
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"


        # Topics 
        action_topic = f"/tb{ros_domain_id}/multiagent/action_id"
        reward_topic = f"/tb{ros_domain_id}/multiagent/reward"


        # publish topics 
        self.action_pub = self.create_publisher(Action, action_topic, 10)

        # subscribe to rewards 
        self.reward_sub = self.create_publisher(Reward, reward_topic, self.reward_callback, 10)

        time.sleep(2)


  
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
        self.action_matrix = np.loadtxt(state_nstate_pth)

        # initialize RL matrix 
        self.RL_matrix = np.zeros((len(self.states), len(self.actions), 2)) # state-action-agenr(3 column matrix)
        self.old_RL_matrix = self.RL_matrix.copy() # convergence 

        # intiatlizing variables 
        self.curr_state_idx = 0 
        self.cur_reward = None 
        self.t = 0 
        # call the algorithm
        self.start_RL_algorithm


    def start_RL_algorithm(self):
        """ Runs the algorithm -- publishes action"""

        # save matrix once Learning has converged 

        max_trj = 500 

        if self.t >= max_trj:  # RL algorithm was has converged 
            self.save_matrix() # save the matrix 

        next_state_idx = random.randint(0, len(self.states))
        
        # use action_matrix (state_nstate) matrix to get a valid next action 
        # because this is simulating the matrix,
        #  don't have to have robots select and combine their actions 

        while self.action_matrix[self.curr_state_idx][next_state_idx] == -1:
            next_state_idx = random.randint(0, len(self.states))

        # get next state and the joint action idx that gets you there 
        self.next_state_idx = next_state_idx 
        self.joint_action_idx = self.action_matrix[self.curr_state_idx][self.next_state_idx]
        self.curr_joint_action = self.actions[self.joint_action_idx]

        self.V_s_prime = None  # resetting minmax value for updating MARL matrix 
        action_msg = Action()
        action_msg.action_id = self.joint_action_idx

        self.action_pub.publish(action_msg) # publish the action 


    def update_matrix(self):
        # given an agent id 
        agent_list  = ['A','D']
        
        for agent_id, _ in agent_list:  # for every agent 
            
            r_i = self.cur_reward[agent_id] 
            # from teh joint reward, extract the agent-specific reward 


            #define model parameters 
            alpha = 1 
            gamma = 1 

            # get random joint actino, by random decison of 
            self.curr_action = self.actions[self.joint_action_idx]

            Q_s_prime_i_flat = self.RL_matrix[self.next_state_idx, :, agent_id] 
            
            # Reshape into the 5x5 Game Matrix M_i (rows=agent_i, columns=opponent)
            M_game_i = Q_s_prime_i_flat.reshape(5,5)

            #Calculate the maxminvalue 
            maxmin = np.max(np.min(M_game_i, axis=1))

            # get the curr value for the next stat e
            old_q_val = self.RL_matrix[self.curr_state_idx, self.joint_action_idx, agent_id]

            # update based on the Minimax Bellman equation
            new_q_agent = old_q_val + alpha * (r_i + gamma * maxmin - old_q_val)
            self.RL_matrix[self.curr_state_idx, self.joint_action_idx, agent_id] = new_q_agent

    
    def reward_callback(self, msg:Reward):
        """ Triggered by reward_simulator publishing a reward, 
        calls an algorithm again after processing the reward, performing RL updates"""
        
        self.cur_reward = msg.reward 
        
        self.update_matrix() # calls fucntion to update the matrix 
        
        self.t += 1  # update number of iterations 

        self.curr_state_idx = self.next_state_idx 
        # set next state as teh curr state to run another action 
        self.start_RL_algorithm() # beign RL algorithm again by getting a new action based on the current state 



    def save_matrix(self):
        """Forces shutdown on Node after q_matrix is saved"""
        matrix_path = os.path.join(self.share, 'matrices', 'RL_matrix.txt')
        np.savetxt(matrix_path, self.RL_matrix)

        self.get_logger().info(f'Saved RL matrix to: {matrix_path}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    # player = sys.argv[1:] # Attack (A), Defend (D)
    node = ReinforceLearn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()