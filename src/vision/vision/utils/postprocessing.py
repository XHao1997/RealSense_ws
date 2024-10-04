import copy
import numpy as np
import cv2
from functools import reduce
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.transform import Rotation as R

def axis_vectors_to_quaternion(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    Convert three axis vectors (x, y, z) representing the orientation into a quaternion.

    Args:
        x_axis (np.ndarray): The x-axis vector (3D).
        y_axis (np.ndarray): The y-axis vector (3D).
        z_axis (np.ndarray): The z-axis vector (3D).

    Returns:
        np.ndarray: A quaternion representing the orientation [x, y, z, w].
    """
    # Ensure the vectors are normalized (unit vectors)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Construct the rotation matrix from the axes
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Convert the rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]

    return quaternion

def find_bbox_center(bbox:list)->list:
    """find bbox center

    Args:
        bbox (list): a list of one bounding box
    Returns:
        list: 4 corner position [x0,y0,x1,y1]

    """
    # find center position of xyxxy bounding box
    return  [int((bbox[0]+bbox[2])/2),
             int((bbox[1]+bbox[3])/2)]
 
def get_bbox_incircle_size(bbox:list):
    """adjust the incircle size base on the width and length
    of the bounding box

    Args:
        bbox (list): _description_

    Returns:
        _type_: _description_
    """
    w = abs(bbox[0]-bbox[2])
    h = abs(bbox[1]-bbox[3])
    return  int(min(w,h)/10)

    
def get_incircle_bbox(img:np.ndarray=None, 
                      bbox_list:list=None):
    """_summary_

    Args:
        img (np.ndarray, optional): _description_. Defaults to None.
        bbox_list (list, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    incircle_list = list(map(find_bbox_center, 
                             bbox_list))
    circle_size = list(map(get_bbox_incircle_size, 
                           bbox_list))
    mask_new = np.zeros_like(img)
    img_new = copy.deepcopy(img)
    # Iterate through the bounding boxes and draw circles
    for center, radius in zip(incircle_list, circle_size):
        cv2.circle(mask_new, center, radius, (255, 255, 255), -1)
        cv2.circle(img_new, center, radius, (255, 255, 255), -1)
        
    result = {}
    result['mask'] = mask_new
    result['image'] = img_new
    result['incircle_list'] = incircle_list
    return result

def get_ROI_image(img:np.ndarray, mask:np.ndarray)->np.ndarray:
    """given origin image and mask, get the ROI image

    Args:
        img (np.ndarray): the original image
        mask (np.ndarray): the mask get from MobileSAM

    Returns:
        np.ndarray: segmented image
    """
    seg_img = np.zeros_like(img)
    seg_img = cv2.bitwise_and(img,img,mask=mask)
    return seg_img

def shrunk_mask(mask:np.ndarray)->np.ndarray:
    """_summary_

    Args:
        mask (np.ndarray): binary mask image (1 channel)

    Returns:
        np.ndarray: shrun mask which is a little smaller
    """
    # Define a kernel (structuring element) for the following operation
    kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel of ones
    # Perform the opening operation to remove noise
    opened_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_OPEN, kernel)
    # Apply erosion to shrink the mask
    shrunk_mask = cv2.erode(opened_img, kernel, iterations=1)
    return shrunk_mask

def get_sam_mask(sam_result)->dict:
    """get the segmentation mask from SAM result

    Args:
        sam_result: result predicted by MobileSAM

    Returns:
        dict: a dictionary including sperate masks and combined
    """
    result = {}
    result['combined_mask'] = reduce(lambda x, y: x + y, sam_result.masks.data.cpu().numpy())
    result['masks'] = sam_result.masks.data.cpu().numpy()
    return result

def visualise_pc(pc,is_numpy=True):
    # Create Open3D PointCloud object
    if is_numpy:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
    else:
        pcd=pc
    # Create a coordinate frame for the origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05, origin=[0, 0, 0])
    # Visualize the point cloud with the origin
    o3d.visualization.draw_geometries([pcd, origin])
    return

def mask_to_pc(mask:np.ndarray, depth_img, depth_scale=0.0010000000474974513)->np.ndarray:
    """_summary_

    Args:
        mask (np.ndarray): _description_
        depth_img (_type_): _description_
        depth_scale (float, optional): _description_. Defaults to 0.0010000000474974513.

    Returns:
        list: _description_
    """
    depth_intrinsic = rs.pyrealsense2.intrinsics()
    depth_intrinsic.width = 640
    depth_intrinsic.height = 480
    depth_intrinsic.ppx = 331.404
    depth_intrinsic.ppy = 239.84
    depth_intrinsic.fx = 607.34
    depth_intrinsic.fy = 607.415
    depth_intrinsic.model = rs.pyrealsense2.distortion.inverse_brown_conrady
    depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    seg_pc=[]
    # Convert the mask to a boolean array (True where mask == 1)
    mask_bool = mask == 1

    # Get the indices of the masked pixels (y, x)
    y_indices, x_indices = np.nonzero(mask_bool)

    # Get the depth values at the masked pixel locations and scale them
    depth_values = depth_img[y_indices, x_indices] * depth_scale

    # Filter out invalid depth values
    valid_depth_mask = (depth_values > 0) & (depth_values < 1)
    valid_y = y_indices[valid_depth_mask]
    valid_x = x_indices[valid_depth_mask]
    valid_depth_values = depth_values[valid_depth_mask]

    # Use rs.rs2_deproject_pixel_to_point to convert pixels to 3D points
    seg_pc = []
    for x, y, depth in zip(valid_x, valid_y, valid_depth_values):
        result = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], depth)
        seg_pc.append(result)
    # Assuming voxel_down_pcd is your point cloud data that you want to downsample and clean
    # Perform statistical outlier removal on the downsampled point cloud
    voxel_down_pcd = o3d.geometry.PointCloud()  # Replace with your actual downsampled point cloud
    voxel_down_pcd.points = o3d.utility.Vector3dVector(seg_pc)  # Use your point data
    # Apply voxel grid downsampling
    voxel_size = 0.00005 # Adjust this value according to the density of your point cloud
    voxel_down_pcd = voxel_down_pcd.voxel_down_sample(voxel_size)
    # Remove statistical outliers
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    return cl

def generate_pc_pose(pc:np.ndarray)->(list,np.ndarray):
    """This function generate the grasp pose from point cloud,
    which includes cenroid point and the orientation pose

    Args:
        pc (np.ndarray): the interest object point cloud

    Returns:
        centroid pont, pose: centroid point of the object and grasp pose
    """
    std_scaler = StandardScaler()
    std_scaler.fit(pc)
    x_std = std_scaler.transform(pc)
    pca = PCA(n_components=3)
    pca.fit_transform(x_std)
    centroid_point = std_scaler.mean_
    pose = pca.components_
    return centroid_point, pose