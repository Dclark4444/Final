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

        self.cmd_publisher = self.create_publisher(Twist, ros_domain_id + '/cmd_vel', 10)


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

        # >>> PLACEHOLDER: your special attack code here <<<

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

