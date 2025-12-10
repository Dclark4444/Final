import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
import numpy as np

from attack_defend_nodes import Attack_move, Defense_move

TURTLE_BOT_ID = '/tb07'

TYPE = input("Press 1 to load defender, press 2 to load attacker ")
if TYPE == '1':
    TYPE = 'Defense'
else:
    TYPE = 'Attack'


class ExecutePolicy(Node):
    def __init__(self):
        super().__init__("execute_policy")

        # load correct NN and moves
        if TYPE == 'Defense':
            self.model = tf.keras.models.load_model("defender.h5")
            self.action_move = Defense_move()
        else:
            self.model = tf.keras.models.load_model("atacker.h5")
            self.action_move = Attack_move()

        self.get_logger().info("Model loaded.")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            TURTLE_BOT_ID + "/camera/image_raw",
            self.callback,
            10
        )

        self.label = None
        self.confidence = None

        # HANIM PUT ARM SUBSCRIBER STUFF

        self.actions = self.action_move.get_action()  # get the actions of the robot

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0][0]

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
    dual_going = True
    while dual_going:
        while node.label == None:
            rclpy.spin(node)
        node.execute()
        node.label = None
        if input('Did the balloon pop y/n?') == 'y':
            dual_going = False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

