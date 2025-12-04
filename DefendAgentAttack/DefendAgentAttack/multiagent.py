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
        action_pth = os.path.join(self.share, 'matrices', 'action.txt')
        self.actions = np.loadtxt(action_pth)

        state_pth = os.path.join(self.share, 'matrices', 'states.txt')
        self.states = np.loadtxt(state_pth)


        # Initialize RL matrix 
        self.RL_matrix = []


        # call the algorithm

    


    def start_RL_algorithm(self):
        """ Runs the algorithm -- publishes action"""

        # save matrix once Learning has converged 

        curr_action = 0 

        self.action_pub.publish(curr_action)


    def reward_callback(self, msg:Reward):
        """ calls an algorithm again after processing the reward """


        self.start_RL_algorithm()



    def save_matrix(self):
        """Forces shutdown on Node after q_matrix is saved"""
        matrix_path = os.path.join(self.share, 'matrices', 'RL_matrix.txt')
        np.savetxt(matrix_path, self.RL_matrix)

        self.get_logger().info(f'Saved RL matrix to: {matrix_path}')
        rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    player = sys.argv[1:] # Attack (A), Defend (D)
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