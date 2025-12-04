import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
import numpy as np

TURTLE_BOT_ID = '/tb04'
IMG_SIZE = (250, 250)

class LiveClassifier(Node):
    def __init__(self):
        super().__init__("live_classifier")

        self.model = tf.keras.models.load_model("left_right_classifier.h5")
        self.get_logger().info("Model loaded.")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            TURTLE_BOT_ID + "/camera/image_raw",
            self.callback,
            10
        )

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0][0]

        label = "RIGHT" if pred >= 0.5 else "LEFT"
        confidence = pred if pred >= 0.5 else 1 - pred

        self.get_logger().info(f"Prediction: {label}   (confidence {confidence:.3f})")

def main(args=None):
    rclpy.init(args=args)
    node = LiveClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
