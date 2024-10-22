#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformStamped
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration

def load_eye_hand(yaml_file_path: str) -> dict:
    """Load camera intrinsics from a YAML file and return the transformation matrix."""
    with open(yaml_file_path, 'r') as yaml_file:
        eye_hand_matrix = yaml.safe_load(yaml_file)
    return eye_hand_matrix

class FramePublisher(Node):
    def __init__(self):
        super().__init__('eye2hand')
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create a tf2 buffer and transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Load transformation matrix from YAML file
        yaml_file_path = 'src/vision/config/eye2hand.yaml'  # Adjust this path to your actual YAML file
        self.eye2hand_matrix = load_eye_hand(yaml_file_path)

        # Attempt to handle the eye-to-hand transformation
        self.handle_eye2hand()

    def handle_eye2hand(self):
        from_frame = 'camera_color_optical_frame'
        to_frame = 'camera_depth_frame'
        try:
            # Look up the transform with a timeout
            future = self.tf_buffer.wait_for_transform_async(to_frame,from_frame,Time())
            rclpy.spin_until_future_complete(self, future)
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                Time(),  # Pass the latest available transform
                timeout=Duration(seconds=1)
            )

            # Print the transformation details
            self.get_logger().info(f"Translation: {transform.transform.translation}")
            self.get_logger().info(f"Rotation: {transform.transform.rotation}")

            # Now publish the eye-to-hand transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'tool0'
            t.child_frame_id = 'camera_color_frame'
            t.transform.translation.x = self.eye2hand_matrix['translation']['x']
            t.transform.translation.y = self.eye2hand_matrix['translation']['y']
            t.transform.translation.z = self.eye2hand_matrix['translation']['z']
            t.transform.rotation.w = self.eye2hand_matrix['rotation']['w']
            t.transform.rotation.x = self.eye2hand_matrix['rotation']['x']
            t.transform.rotation.y = self.eye2hand_matrix['rotation']['y']
            t.transform.rotation.z = self.eye2hand_matrix['rotation']['z']
            self.tf_static_broadcaster.sendTransform(t)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"Failed to find transform: {str(e)}")

def main():
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
