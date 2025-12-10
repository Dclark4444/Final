import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
import numpy as np

from '' import poses

TURTLE_BOT_ID = '/tb07'

TYPE = input("Press 1 to load defender, press 2 to load attacker ")
if TYPE == '1':
    TYPE = 'defender'
else:
    TYPE = 'attacker'

class ExecutePolicy(Node):
    def __init__(self):
        super().__init__("execute_policy")

        #load correct NN
        if TYPE == 'defender':
            self.model = tf.keras.models.load_model("defender.h5")
        else:
            self.model = tf.keras.models.load_model("atacker.h5")
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

        #HANIM PUT ARM SUBSCRIBER STUFF

        self.actions = poses.get_actions(TYPE)

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

def main(args=None):
    rclpy.init(args=args)
    node = ExecutePolicy()
    duel_going = True
    while duel_going:

        #wait until read & prediction
        while self.label == None:
            rclpy.spin(node)
        #do the action based on the prediction and then reset
        self.actions[self.label]

        #check if the baloon popped, this way the bots require our input to continue
        if input("Did the ballloon pop? y/n") == 'y':
            duel_going = False

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()