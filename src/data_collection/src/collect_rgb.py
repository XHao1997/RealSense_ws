#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
import os
from datetime import datetime
import rospkg
rp = rospkg.RosPack()
class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        
        # Declare and get the parameter for the image topic
        self.declare_parameter('image_topic', 'camera/camera/color/image_raw')
        self.declare_parameter('telekey_topic', '/turtle1/cmd_vel')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value            
        telekey_topic = self.get_parameter('telekey_topic').get_parameter_value().string_value
        # Subscribe to the image and telekey topics
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_subscription(Twist, telekey_topic, self.telekey_callback, 10)
        
        # Initialize variables
        self.cv_image = None
        self.image_save_directory = os.getcwd()+'/src/data'
        if not os.path.exists(self.image_save_directory):
            os.makedirs(self.image_save_directory)
        
        self.get_logger().info('Image Saver Node has been started.')
        self.get_logger().info(f'Image will be saved in {self.image_save_directory}')

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {str(e)}")

    def telekey_callback(self, msg: Twist):
        if msg is not None:
            self.get_logger().info(f"Key Pressed!")
            if self.cv_image is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.image_save_directory, f"image_{timestamp}.png")
                cv2.imwrite(filename, self.cv_image)
                self.get_logger().info(f"Image saved: {filename}")
            else:
                self.get_logger().warn("No image received yet to save.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
