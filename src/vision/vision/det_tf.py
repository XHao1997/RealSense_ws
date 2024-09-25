# Subscribe Detection result and broadcast tf
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformStamped
from geometry_msgs.msg import Point
from ultralytics_ros.msg import YoloResult

class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')
        self.subscription = self.create_subscription(
            YoloResult,
            'yolo_result',
            self.det_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.tf_broadcaster = TransformBroadcaster(self)

    def det_callback(self, msg:YoloResult):
        objs = {}
        detections = msg.detections.detections
        self.get_logger().info(f'My log message {detections}', 
                               skip_first=True, throttle_duration_sec=1.0)

        for i in range(len(detections)):
            objs[detections[i].results[0].hypothesis.class_id+f"{i}"] = detections[i].bbox.center.position


        self.get_logger().info(f'My log message {objs}', 
                               skip_first=True, throttle_duration_sec=1.0)
        # np.save('det_msg',msg.detections)

        for name, pos in objs.items():
            pos:Point
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'
            t.child_frame_id = name

            t.transform.translation.x = pos.x/1000
            t.transform.translation.y = pos.y/1000
            t.transform.translation.z = 0.10
            self.get_logger().info(f'My log message {pos}', skip_first=True, throttle_duration_sec=1.0)
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    det_tf = DetTF()

    try:
        rclpy.spin(det_tf)
    except KeyboardInterrupt:
        pass

    det_tf.destroy_node()
    rclpy.shutdown()