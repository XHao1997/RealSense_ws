#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TelekeyPublisher(Node):
    def __init__(self):
        super().__init__('telekey_publisher')
        self.publisher_ = self.create_publisher(String, 'telekey_topic', 10)
        self.get_logger().info('Telekey Publisher Node has been started.')

    def run(self):
        try:
            while rclpy.ok():
                # Check for key press
                key = input("Press 's' to save image: ")
                if key == 's':
                    msg = String()
                    msg.data = 'save_image'
                    self.publisher_.publish(msg)
                    self.get_logger().info('Published save_image command.')
        except KeyboardInterrupt:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = TelekeyPublisher()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
