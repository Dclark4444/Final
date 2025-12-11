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


class MovementNode(Node, ABC):
    """ Superclass containing all shared attributes, initializations, and helper methods. """

    def __init__(self, node_name):
        super().__init__(node_name)

        # 2. Shared Data/RL parameters
        self.curr_at = None
        self.agent_id = None
        self.go_act = None

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

        self.actions = {'Attack': {'Left', 'Right'},
                        'Defense': {'Left', 'Right'}}

        # 4. Shared Timing/Action Variables
        self.action_MIN_RUN_TIME = 3  # seconds  Move and reverse back
        self.start_time = None

        # 5. Shared home pose
        self.home_pose = [0.0, 0.0, 0.0, 0.0]
        self.scan_dist = None
        self.desired_scan_dist = 0.3

        # diff variables for Attack and Defend
        self.G_R_arm_pose = []
        self.desired_angle = 0.0
        self.angvelocity = 0.0
        self.desired_dist = 0.0
        self.linear_v = 0.0

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # actions are 3 for defense:
        # Guard Left, Guard Right and Reverse

        # Topics
        cmd_vel_topic = f"/tb{ros_domain_id}/cmd_vel"
        # compress_image_topic = f'/tb{self.ros_domain_id}/oakd/rgb/preview/image_raw/compressed'
        scan_topic = f'/tb{self.ros_domain_id}/scan'
        arm_grip = f'/tb{ros_domain_id}/target_gripper_position'
        arm_angles = f'/tb{ros_domain_id}/target_joint_angles'

        # publishing
        self.joint_arm_pub = self.create_publisher(ArmJointAngles, arm_angles, 10)
        self.gripper_pub = self.create_publisher(ArmGripperPosition, arm_grip, 10)
        self.movement_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic,
                                                 self.scan_callback, 10)

        self.timer_period = 0.25

    def get_action(self):
        self.actions = self.actions[self.agent_id]

    def body_movement(self, neg_velo, forward=False):  # if velocity is negative then it will turn left
        x = -1 if neg_velo else 1
        twist_msg = Twist()
        twist_msg.angular.z = x * self.angvelocity  # tbh i don't know which way is left or not

        if forward:
            twist_msg.linear.x = self.linear_v
        self.movement_publisher.publish(twist_msg)

    def arm_move(self, left_joint_1, pose):
        # if joint angle negative ==== Guarding left
        x = -1 if left_joint_1 else 1
        arm_msg = ArmJointAngles()
        arm_msg.joint1 = x * pose[0]
        arm_msg.joint2, arm_msg.joint3, arm_msg.joint4 = pose[1:3]
        self.joint_arm_pub.publish(arm_msg)

    def scan_callback(self, scan: LaserScan):  # reads scan images, only when action is performed.
        self.get_logger().debug('LaserScan received')
        scan_ranges = np.array(scan.ranges)
        num_points = len(scan_ranges)

        # Define slices for forward direction
        window = max(5, num_points // 25)

        forward_slice = np.concatenate((scan_ranges[:window], scan_ranges[-window:]))

        valid = forward_slice[np.isfinite(forward_slice)]
        if len(valid) == 0:
            self.get_logger().warn("No valid scan points in front.")
            return

        self.scan_dist = float(np.median(valid))  # forward distance

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # first instance of 3.5 second run, go to the next action, after accesing state
        if elapsed > self.action_MIN_RUN_TIME:
            self.get_logger().info("Action should have finished performing")

            if (self.scan_dist - self.desired_scan_dist) < 0.5:
                rclpy.shutdown()  ## immediate shutdown if the robots are very close to eachother

            self.start_time = None
            if self.go_act == 1:  # run teh reverse once there is a singal that the action has occured
                self.run_action(self.curr_at, reverse=True)  # run action in reverse after elapsed time, restart clock
            # else do nothing

    def policy_loop(self):
        self.run_action(self.curr_at)

    @abstractmethod
    def run_action(self):
        raise NotImplementedError("run_action method must be implemented by the child class (Defense/Attack).")


class Attack_move(MovementNode):
    def __init__(self):
        super().__init__('attack_move_nn')

        self.agent_id = 'Attack'

        # tie variables for performing actions (limit)
        self.action_MIN_RUN_TIME = 2  # seconds
        self.start_time = None

        # self.step = "Accessing"

        self.arm_going_left = [0.530, 0, 0.320, 0]  # GIve this values

        self.desired_angle = 60  ### DESIRED ANGLE IS THE ONLY THING TattacHAT CHANGED
        self.angvelocity = self.desired_angle / (self.action_MIN_RUN_TIME)

        self.desired_dist = 0.3  # meters
        self.linear_v = self.desired_dist / self.action_MIN_RUN_TIME
        # THis is arm pose for R but Left is negative of the first joint

        self.timer = self.create_timer(self.timer_period, self.policy_loop)

    def run_action(self, action, reverse=False):

        self.curr_at = action

        if self.start_time != None:
            self.start_time = self.get_clock().now()

        if reverse:
            # reverse the action you just did
            if self.curr_at == 'Left':
                self.curr_at = 'Right'
            else:
                self.curr_at = 'Left'

        if self.curr_at == 'Left':
            # complete 30 dtegree movement in 2.0 seconds # negative velocity when going left
            self.body_movement(neg_velo=True,
                               forward=True)  # move (+ velocity)  and turn with a ngative angular velocity
            self.arm_move(False, self.A_R_arm_pose)  # move arm to teh right of the robot

        if self.curr_at == 'Right':
            self.body_movement(neg_velo=False,
                               forward=True)  # move ( + velocity) and turn with a positive angular velocity
            self.arm_move(True, self.A_R_arm_pose)  # move arm down to thte right


class Defense_move(MovementNode):
    def __init__(self):
        super().__init__('defense_move_nn')

        self.agent_id = 'Defense'
        self.action_MIN_RUN_TIME = 2  # seconds
        self.start_time = None

        self.G_R_arm_pose = [1.33, 0.4, 0.2, 0.8]
        self.desired_angle = 30
        self.angvelocity = self.desired_angle / (self.action_MIN_RUN_TIME)

        self.timer = self.create_timer(self.timer_period, self.policy_loop)

    def run_action(self, action, reverse=False):

        self.curr_at = action
        if self.start_time != None:
            self.go_act = 1
            self.start_time = self.get_clock().now()

        if reverse:
            # reverse the action you just did
            if self.curr_at == 'Left':
                self.curr_at = 'Right'
            else:
                self.curr_at = 'Left'

        if self.curr_at == 'Left':
            # complete 30 dtegree movement in 2.0 seconds
            # turn left and move arm to negative angular velocity
            self.body_movement(neg_velo=True)  # turn body in negative direction ,
            # move arm to teh peft
            self.arm_move(True, self.G_R_arm_pose)

        if self.curr_at == 'Right':
            self.body_movement(neg_velo=False)  # turn right ( positive angualr velocity)
            self.arm_move(False, self.G_R_arm_pose)  # move arm to teh right as well


