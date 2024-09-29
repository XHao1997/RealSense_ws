# Subscribe Detection result and broadcast tf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster, TransformStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from message_filters import TimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from .utils import postprocessing
from ultralytics_ros.msg import YoloResult
import ultralytics
import cv_bridge
import numpy as np
class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')

        self.declare_parameter("sam_topic", "sam_result")
        self.declare_parameter("depth_topic", "camera/camera/aligned_depth_to_color/image_raw")
        # define input/output topic 
        sam_topic = (
            self.get_parameter("sam_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        
        self.sub1 = Subscriber(self, Image, sam_topic)
        self.sub2 = Subscriber(self, Image, depth_topic)

        # ApproximateTimeSynchronizer will call the callback only when both messages are available
        self.ts = ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=1, slop=0.5)
        self.ts.registerCallback(self.det_callback)
        self.bridge = cv_bridge.CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info('Publishing1: ""')
        self.results_pub = self.create_publisher(Image, "det", 10)
        

    def det_callback(self, msg1:Image, msg2:Image):
        
        depth_img = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='bgr8')
        mask = self.bridge.imgmsg_to_cv2(msg2, desired_encoding='passthrough')
        if np.max(mask)!=0:
            self.get_logger().info("hi")
            pc = postprocessing.mask_to_pc(mask, depth_img)
            centroid_point, ori = postprocessing.generate_pc_pose(pc)
        
        # if len(pc)!=0:
        #     centroid_point, ori = postprocessing.generate_pc_pose(pc)
        #     self.get_logger().info(centroid_point)
        
        # self.results_pub.publish(msg2)
        return

def main(args=None):
    rclpy.init(args=args)
    det = DetTF()
    rclpy.spin(det)
    # Destroy the node explicitly
    # optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    det.destroy_node()
    rclpy.shutdown()