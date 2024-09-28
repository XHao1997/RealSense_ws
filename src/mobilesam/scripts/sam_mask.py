#!/usr/bin/env python3
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics_ros.msg import YoloResult
from ultralytics import SAM
import numpy as np 
from message_filters import Subscriber, ApproximateTimeSynchronizer
import ros2_numpy as rnp
from mobilesam import postprocessing
import cv2
import matplotlib.pyplot as plt

class SAM_Node(Node):

    def __init__(self):
        super().__init__('SAM_node')
        self.declare_parameter("sam_model", "sam_l.pt")
        self.declare_parameter("image_topic", "camera/camera/color/image_raw")
        self.declare_parameter("yolo_topic", "yolo_result")
        self.declare_parameter("result_topic", "sam_result")
        sam_model = self.get_parameter("sam_model").get_parameter_value().string_value
        # define input/output topic 
        input_topic_1 = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        input_topic_2 = (
            self.get_parameter("yolo_topic").get_parameter_value().string_value
        )
        result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )
        # Create subscribers for topic_1 and topic_2
        self.subscriber_1 = Subscriber(self, Image, input_topic_1)
        self.subscriber_2 = Subscriber(self, YoloResult, input_topic_2)
        self.model = SAM(f"src/ultralytics_ros/weights/{sam_model}")
        # ApproximateTimeSynchronizer will call the callback only when both messages are available
        self.ts = ApproximateTimeSynchronizer([self.subscriber_1, self.subscriber_2], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.subscriber_callback)
        self.results_pub = self.create_publisher(Image, result_topic, 5)
        self.bridge = cv_bridge.CvBridge()
        
    def subscriber_callback(self, msg1, msg2):
        objs = {}
        # Convert the ROS image (msg1) to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        detections = msg2.detections.detections
        for i in range(len(detections)):
            pose_list = [detections[i].bbox.center.position,detections[i].bbox.size_x,detections[i].bbox.size_y]
            objs[detections[i].results[0].hypothesis.class_id+f"{i}"]= pose_list
            
        sam_mask = np.zeros([480, 640]).astype(np.uint8)
        sam_pose = []
        sam_bbox = []
        for name, pos in objs.items():
            sam_pose.append([pos[0].x,pos[0].y])
            sam_bbox.append(pose2d_to_xyxy_bbox(pos[0],pos[1],pos[2]))
        if len(sam_pose)!=0:
            # Run SAM model on the converted image
            sam_result = self.model.predict(cv_image, points=sam_pose,bboxes=sam_bbox)
            # Extract the SAM mask (assuming 'combined_mask' is in the result)
            sam_mask = postprocessing.get_sam_mask(sam_result)['combined_mask']
            sam_mask = sam_mask.astype(np.uint8)
            sam_mask = postprocessing.shrunk_mask(sam_mask)
        # Ensure the mask is of type uint8 for further processing
        # Convert single-channel mask to 3-channel RGB if necessary
        if len(sam_mask.shape) == 2:
            sam_mask_rgb = cv2.cvtColor(sam_mask, cv2.COLOR_GRAY2RGB)
        else:
            sam_mask_rgb = sam_mask
        covered_img = cv2.bitwise_and(cv_image,cv_image,mask=sam_mask)

        # Convert the mask back to a ROS Image message
        msg_mask = self.bridge.cv2_to_imgmsg(covered_img, encoding="bgr8")
        self.results_pub.publish(msg_mask)

def pose2d_to_xyxy_bbox(pose, width, height):
    # Extract the center from the Pose2D
    x_center, y_center = pose.x, pose.y

    # Calculate xmin, ymin, xmax, ymax
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    # Return bbox in xyxy format (xmin, ymin, xmax, ymax)
    return [xmin, ymin, xmax, ymax]

def main(args=None):
    rclpy.init(args=args)
    sam_sub = SAM_Node()
    rclpy.spin(sam_sub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()