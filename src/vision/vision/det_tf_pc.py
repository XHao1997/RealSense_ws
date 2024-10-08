# Subscribe Detection result and broadcast tf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
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
import time
import std_msgs
from contextlib import contextmanager
import yaml
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

@contextmanager
def timer():
    start_time = time.time()
    yield
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print elapsed time formatted to three decimal places
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
def load_camera_intrinsics(yaml_file_path: str) -> rs.intrinsics:
    """Load camera intrinsics from a YAML file and return an rs.intrinsics object.

    Args:
        yaml_file_path (str): Path to the YAML file containing camera intrinsics.

    Returns:
        rs.intrinsics: An instance of rs.intrinsics populated with the values from the YAML file.
    """
    # Load intrinsics from the YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        intrinsics = yaml.safe_load(yaml_file)

    # Access the intrinsic values
    intrinsic = intrinsics['camera_intrinsics']

    # Create a RealSense intrinsics object
    color_intrinsic = rs.intrinsics()
    color_intrinsic.width = intrinsic['width']
    color_intrinsic.height = intrinsic['height']
    color_intrinsic.ppx = intrinsic['ppx']
    color_intrinsic.ppy = intrinsic['ppy']
    color_intrinsic.fx = intrinsic['fx']
    color_intrinsic.fy = intrinsic['fy']
    color_intrinsic.model = rs.distortion.inverse_brown_conrady  # Assuming you want this model
    color_intrinsic.coeffs = intrinsic['coeffs']

    return color_intrinsic

class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')

        self.declare_parameter("sam_topic", "sam_result")
        self.declare_parameter("depth_topic", "sam_depth")
        self.declare_parameter("cam_depth_topic", "camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("yolo_topic", "yolo_result")
        self.declare_parameter("use_sam", False)
        self.declare_parameter("pc_topic", 'camera/camera/depth/color/points')
        
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
        pc_topic = (
            self.get_parameter("pc_topic").get_parameter_value().string_value
        )

        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.yolo_sub = Subscriber(self, YoloResult, yolo_topic)
        self.cam_depth_sub = Subscriber(self, Image, cam_depth_topic)
        # Subscribe to PointCloud2 topic
        self.pc_sub = Subscriber(self, PointCloud2, pc_topic)
        yaml_file_path = 'src/vision/config/camera_intrinsics.yaml'  # Change this to your actual YAML file path
        self.color_intrinsic = load_camera_intrinsics(yaml_file_path)

        self.get_logger().info("using yolo-seg")
        # define the sub topic list
        sub_list = [self.cam_depth_sub, self.yolo_sub, self.pc_sub]
        
        self.ts = TimeSynchronizer(sub_list, queue_size=1000)
        self.ts.registerCallback(self.det_callback_yolo)
            
        self.bridge = cv_bridge.CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.publisher_ = self.create_publisher(PointCloud2, 'point_cloud_topic', 10)

    @staticmethod
    def convert_pointcloud2_to_numpy(cloud_msg):
        # Convert the point cloud to a generator
        point_cloud_gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Convert the generator to a numpy array (XYZ values only)
        points = np.fromiter(point_cloud_gen, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        
        # Stack the structured array into a regular NumPy array (n x 3)
        points_array = np.vstack((points['x'], points['y'], points['z'])).T

        return points_array
    
    @staticmethod
    def convert_numpy_to_open3d(np_points):
        # Create an Open3D point cloud object
        o3d_cloud = o3d.geometry.PointCloud()
        
        # Set Open3D point cloud points from numpy array
        o3d_cloud.points = o3d.utility.Vector3dVector(np_points[:, :3])  # Take the XYZ part
                # Downsample the point cloud using voxel size
        voxel_size = 0.001  # Adjust the voxel size as needed
        # downsampled_cloud = o3d_cloud.voxel_down_sample(voxel_size)
        
        return o3d_cloud

    def det_callback_yolo(self,  msg1:Image, msg2:YoloResult, msg3:PointCloud2):
        depth_img = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='passthrough')
        mask_list = []

        for mask in msg2.masks:
            # Convert the ROS Image message to an OpenCV image
            post_mask = postprocessing.shrunk_mask(self.bridge.imgmsg_to_cv2(mask,desired_encoding='passthrough'),cut_incircle=True)
            mask_list.append(post_mask)
        if msg2.detections.detections: # if there is detection
            bbox_centers = [[det.bbox.center.position.x, det.bbox.center.position.y] for det in msg2.detections.detections]
            bbox_centers_3d = [postprocessing.pixel_to_xyz(bbox_center, depth_img, self.color_intrinsic) for bbox_center in bbox_centers]
            # Convert the NumPy array to a string and log it
            combined_mask = postprocessing.combine_mask_list(mask_list)
            # combined_mask = self.bridge.imgmsg_to_cv2(combined_mask, desired_encoding='passthrough')
            pcd = self.convert_pointcloud2_to_numpy(msg3)
            pcd = self.convert_numpy_to_open3d(pcd)
            # Build KDTree for nearest neighbor search
            kdtree = postprocessing.create_kdtree(pcd)
            search_num = postprocessing.find_valid_pixel_count(mask_list)
            centroid_list = []
            pose_list = []
            pcd_idx = postprocessing.find_nearest_k_points(kdtree, bbox_centers_3d, k=search_num)
            for i, valid_num in enumerate(search_num):
                if np.max(pcd_idx[i])<len(pcd.points):
                    centroid, pose = postprocessing.generate_pc_pose(np.asarray(pcd.points)[pcd_idx[i]])
                    self.get_logger().debug(f'Logging NumPy Array: \n{pcd_idx}')
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
                    
            next_centoid = np.mean(np.asarray(centroid_list).reshape(-1,3),axis=0)
            next_veiw = np.mean(np.asarray(pose_list).reshape(-1,4),axis=0)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'
            t.child_frame_id = 'next view'
            q = next_veiw
            t.transform.translation.x = next_centoid[0]
            t.transform.translation.y = next_centoid[1]
            t.transform.translation.z = next_centoid[2]
            t.transform.rotation.w = q[3]
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            self.tf_broadcaster.sendTransform(t)
            # Publish the point cloud message
            # Convert NumPy array to PointCloud2 message
            # Create the PointCloud2 message header
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()  # Set the current time
            header.frame_id = "camera_color_optical_frame"  # Change this to your relevant frame
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, np.asarray(pcd.points).tolist())
            self.publisher_.publish(point_cloud_msg)
            self.get_logger().info("Published point cloud.")
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