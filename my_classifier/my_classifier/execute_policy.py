import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import tensorflow as tf
import numpy as np

import time

from my_classifier.attack_defend_nodes import Attack_move, Defense_move

IMG_SIZE = (250, 250)

TYPE = input("Press 1 to load defender, press 2 to load attacker ")
if TYPE == '1':
    TYPE = 'Defense'
else:
    TYPE = 'Attack'


class ExecutePolicy(Node):
    def __init__(self):
        super().__init__("execute_policy")

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

        # load correct NN and moves
        if TYPE == 'Defense':
            self.model = tf.keras.models.load_model("defender.h5")
            self.action_move = Defense_move()
        else:
            self.model = tf.keras.models.load_model("attacker.h5")
            self.action_move = Attack_move()

        self.get_logger().info("Model loaded.")

        self.bridge = CvBridge()
        self.image = None

        self.camera = self.create_subscription(
            Image,
            ros_domain_id + "/oakd/rgb/preview/image_raw",
            self.callback,
            10
        )

        self.label = None
        self.confidence = None

        self.actions = self.action_move.get_action()
        self.get_logger().info('completed init. process')

    def callback(self, msg):
        try:
            self.get_logger().info(f"0")
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                self.get_logger().info(f"frame failed")
            self.get_logger().info(f"1")
            img = cv2.resize(frame, IMG_SIZE)
            self.get_logger().info(f"2")
            img = img.astype(np.float32) / 255.0
            self.get_logger().info(f"3")
            self.image = np.expand_dims(img, axis=0)
        except:
            self.get_logger().info(f"Camera could not be read")
            self.image = None

    def predict(self):
        self.get_logger().info('pre-prediction')
        pred = self.model.predict(self.image, verbose=0)[0][0]
        self.get_logger().info('post-prediction')

        self.label = "RIGHT" if pred >= 0.5 else "LEFT"
        self.confidence = pred if pred >= 0.5 else 1 - pred

        self.get_logger().info(f"Prediction: {label}   (confidence {confidence:.3f})")

    def execute(self):
        # execute the action
        # self.label is the action. -- pass the action in run_action
        self.action_move.run_action(self.label)


def main(args=None):
    rclpy.init(args=args)
    # player = sys.argv[1:] # Attack (A), Defend (D)
    # node = ExecuteOptimal(player)
    node = ExecutePolicy()
    time.sleep(2)
    dual_going = True
    while dual_going:
        while node.image is None:
            node.get_logger().warn('SEARCHING')
            rclpy.spin_once(node, timeout_sec=0.1)
        node.predict()
        node.execute()
        node.label = None
        if input('Did the balloon pop y/n?') == 'y':
            dual_going = False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

