import os
import rclpy
from rclpy import Node

from AttackDefend_interfaces.msg import Reward
from AttackDefend_interfaces.msg import Action 
from std_msgs.msg import Header 


class LearningMatrix(Node):
    def __init__(self):
        super().__init__('simulator for RL')


        #different types of rewards
        self.attack_reward = 50  # large reward for popping the balloon (for attacker)
        self.defend_reward = -5   # small consequence for not being in front of the robot
        self.non_reward = 0 


        # goal states for attacker / defender: 
        # goal of attacker is negative reward of defender 
        self.goal_attack = []
        self.goal_defend = []

        # initialized state 
        self.curr_state = []

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



    def send_reward(self, msg:Action): 
# reward function for Reinforcement Learning 

        # load message (logger)


        # identify the curr_state 


        # set default reward 


        # get the reward based on the state 

        None 



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


        