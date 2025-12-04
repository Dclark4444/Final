#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class SingleFrameCapturer(Node):
    def __init__(self):
        super().__init__('single_frame_capturer')

        # Directory to save images
        self.image_dir = "/home/clarkd26/intro_robo_ws/src/final_project/final_project/images/"
        os.makedirs(self.image_dir, exist_ok=True)

        # Counter for saved images
        self.counter = self._get_next_image_number()

        self.bridge = CvBridge()
        self.latest_frame = None

        # Subscribe to the TurtleBot4 camera
        self.sub = self.create_subscription(
            Image,
            '/tb07/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info("Ready. Press ENTER to capture an image...")

    def _get_next_image_number(self):
        """Find the next available file index."""
        existing = [int(f.split('.')[0]) for f in os.listdir(self.image_dir)
                    if f.endswith(".png") and f.split('.')[0].isdigit()]
        return max(existing) + 1 if existing else 1

    def image_callback(self, msg):
        """Store the most recent camera frame."""
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def capture_and_save(self):
        """Save the latest frame with an incremented filename."""
        if self.latest_frame is None:
            self.get_logger().warn("No frame received yet. Try again.")
            return

        filename = f"{self.counter}.png"
        filepath = os.path.join(self.image_dir, filename)

        cv2.imwrite(filepath, self.latest_frame)
        self.get_logger().info(f"Saved image: {filepath}")

        self.counter += 1


def main():
    rclpy.init()
    node = SingleFrameCapturer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # Wait for ENTER
            key = input("")
            node.capture_and_save()
            print("Press ENTER to take another picture...")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

