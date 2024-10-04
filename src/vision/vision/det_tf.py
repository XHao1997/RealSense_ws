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
import cv2
class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')

        self.declare_parameter("sam_topic", "sam_result")
        self.declare_parameter("depth_topic", "sam_depth")
        self.declare_parameter("cam_depth_topic", "camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("yolo_topic", "yolo_result")
        self.declare_parameter("use_sam", False)
        use_sam = (
            self.get_parameter("use_sam").get_parameter_value().string_value
        )
        # define input/output topic 
        sam_topic = (
            self.get_parameter("sam_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        yolo_topic = (
            self.get_parameter("yolo_topic").get_parameter_value().string_value
        )
        cam_depth_topic = (
            self.get_parameter("cam_depth_topic").get_parameter_value().string_value
        )

        self.sam_sub = Subscriber(self, Image, sam_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.yolo_sub = Subscriber(self, YoloResult, yolo_topic)
        self.cam_depth_sub = Subscriber(self, Image, cam_depth_topic)
        
        if use_sam:
            self.get_logger().info("using SAM-seg")
            sub_list = [self.sam_sub, self.depth_sub, self.yolo_sub]
            # ApproximateTimeSynchronizer will call the callback only when both messages are available
            self.ts = ApproximateTimeSynchronizer(sub_list, slop=0.25, queue_size=1000)
            self.ts.registerCallback(self.det_callback_SAM)
        else:
            self.get_logger().info("using yolo-seg")
            sub_list = [self.cam_depth_sub, self.yolo_sub]
            self.ts = ApproximateTimeSynchronizer(sub_list, queue_size=50, slop=0.001)
            self.ts.registerCallback(self.det_callback_yolo)
            
        self.bridge = cv_bridge.CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        

    def det_callback_SAM(self, msg1:Image, msg2:Image, msg3:YoloResult):
        mask = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='passthrough')[:,:,0]
        depth_img = self.bridge.imgmsg_to_cv2(msg2, desired_encoding='passthrough')
        n = 0
        if np.max(mask)!=0:
            self.get_logger().info("hi")
            pcd =postprocessing.mask_to_pc(mask, depth_img)
            if len(pcd.points)!=0:      
                postprocessing.generate_pc_pose(np.asarray(pcd.points))
                # Generate the pose (centroid and orientation) from the point cloud
                centroid, pose = postprocessing.generate_pc_pose(np.asarray(pcd.points))
                self.get_logger().info(f'{centroid[0]}')
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'camera_color_optical_frame'
                t.child_frame_id = "leaf"
                q = postprocessing.axis_vectors_to_quaternion(pose[0],pose[1],pose[2])
                t.transform.translation.x = centroid[0]
                t.transform.translation.y = centroid[1]
                t.transform.translation.z = centroid[2]
                t.transform.rotation.w = q[3]
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                self.tf_broadcaster.sendTransform(t)
        return
    def det_callback_yolo(self,  msg1:Image, msg2:YoloResult):
        depth_img = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='passthrough')

        if len(msg2.detections.detections)>0:
            centroid_list = []
            pose_list = []
            for i, mask in enumerate(msg2.masks):
                mask = self.bridge.imgmsg_to_cv2(mask, desired_encoding='passthrough')
                # if np.max(mask)!=0:
                pcd =postprocessing.mask_to_pc(mask, depth_img)
                self.get_logger().info(f'{len(pcd.points)}')
                if len(pcd.points)!=0:      
                    # Generate the pose (centroid and orientation) from the point cloud
                    centroid, pose = postprocessing.generate_pc_pose(np.asarray(pcd.points))
 
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = 'camera_color_optical_frame'
                    try:
                        t.child_frame_id = msg2.detections.detections[i].results[0].hypothesis.class_id
                    except IndexError:
                        print("Oops!  That was no valid number. index=",i)
                    q = postprocessing.axis_vectors_to_quaternion(pose[0],pose[1],pose[2])
                    centroid_list.append(centroid)
                    pose_list.append(q)
                    t.transform.translation.x = centroid[0]
                    t.transform.translation.y = centroid[1]
                    t.transform.translation.z = centroid[2]
                    t.transform.rotation.w = q[3]
                    t.transform.rotation.x = q[0]
                    t.transform.rotation.y = q[1]
                    t.transform.rotation.z = q[2]
                    self.tf_broadcaster.sendTransform(t)
            next_centoid = np.mean(np.asarray(centroid_list),axis=0)
            next_veiw = np.mean(np.asarray(pose_list),axis=0)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'
            t.child_frame_id = 'next view'
            if len(next_centoid)!=0:
                q = next_veiw
                t.transform.translation.x = next_centoid[0]
                t.transform.translation.y = next_centoid[1]
                t.transform.translation.z = next_centoid[2]
                t.transform.rotation.w = q[3]
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                self.tf_broadcaster.sendTransform(t)
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