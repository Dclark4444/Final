import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import math
import tensorflow as tf
import numpy as np

import time
from geometry_msgs.msg import Twist
from omx_cpp_interface.msg import ArmGripperPosition, ArmJointAngles


TURN_SPEED = 0.6
FORWARD_SPEED = 0.2
TURN_ANGLE = math.radians(30)
FORWARD_DIST = 0.5


class Movement(Node):
    def __init__(self):
        super().__init__('movement')

        # get the ROS_DOMAIN_ID aka robot number
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "/tb0" + str(int(ros_domain_id))
            else:
                ros_domain_id = "/tb" + str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')


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



        # Arm poses 
        self.home_pose = [0.0, 0.0, 0.0, 0.0]
        self.arm_go_right = [0.530, 0.150, -0.25, -0.5] #this is going in the right direction 





    def arm_move(self, pose, left_joint_1=False):
        # if joint angle negative ==== Guarding left ==> moving Left 
        x = -1 if left_joint_1 else 1
        arm_msg = ArmJointAngles()
        arm_msg.joint1 = x * pose[0]
        arm_msg.joint2, arm_msg.joint3, arm_msg.joint4 = pose[1:3]
        self.joint_arm_pub.publish(arm_msg)


    def attack_left(self):
        cmd = Twist()
        cmd.angular.z = TURN_SPEED
        duration = TURN_ANGLE / TURN_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.angular.z = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        cmd.linear.x = FORWARD_SPEED
        duration = FORWARD_DIST / FORWARD_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.linear.x = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        #assming that attack_let mens that the arm moves left 
        self.arm_move(self.arm_go_right, left_joint_1=True)

        cmd.linear.x = -FORWARD_SPEED
        duration = FORWARD_DIST / FORWARD_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.linear.x = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        cmd.angular.z = -TURN_SPEED
        duration = TURN_ANGLE / TURN_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.angular.z = 0.0
        self.cmd_publisher.publish(cmd)

    def attack_right(self):
        cmd = Twist()

        cmd.angular.z = -TURN_SPEED
        duration = TURN_ANGLE / TURN_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.angular.z = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        cmd.linear.x = FORWARD_SPEED
        duration = FORWARD_DIST / FORWARD_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.linear.x = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        # >>> PLACEHOLDER: your special attack code here <<<
        
        #ssming that attack_let mens that the arm moves left (vice verse) 
        self.arm_move(self.arm_go_right, left_joint_1=False)

        cmd.linear.x = -FORWARD_SPEED
        duration = FORWARD_DIST / FORWARD_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.linear.x = 0.0
        self.cmd_publisher.publish(cmd)
        time.sleep(0.2)

        cmd.angular.z = TURN_SPEED
        duration = TURN_ANGLE / TURN_SPEED
        self.cmd_publisher.publish(cmd)
        time.sleep(duration)

        cmd.angular.z = 0.0
        self.cmd_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = Movement()
    while True:
        if input() == "l":
            node.attack_left()
        else:
            node.attack_right()
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    rclpy.shutdown()

