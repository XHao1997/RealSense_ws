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
from scipy.spatial import KDTree

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
    # Check if the Z-axis is facing toward the frame (positive Z direction)
    if z_axis[2] < 0:
        # Flip the Z-axis
        z_axis = -z_axis
        # To maintain a right-handed coordinate system, flip the Y-axis as well
        y_axis = -y_axis
        x_axis = -x_axis
    
    # Check if the X-axis is facing left (negative X direction)
    if x_axis[0] < 0:
        
        # Flip the Z-axis
        x_axis = -x_axis
        # To maintain a right-handed coordinate system, flip the Y-axis as well
        y_axis = -y_axis
        
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

def pixel_to_xyz(pixel, depth_img, color_intrinsic, depth_scale=0.0010000000474974513):
    # Filter out invalid depth values
    depth_values = depth_img*depth_scale
    x, y = int(pixel[0]), int(pixel[1])
    depth = depth_values[y, x]
    # Get the indices of the masked pixels (y, x)
    # Use rs.rs2_deproject_pixel_to_point to convert pixels to 3D points
    result = rs.rs2_deproject_pixel_to_point(color_intrinsic, [x, y], depth)
    return result
    
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

def shrunk_mask(mask:np.ndarray, iterations=1, cut_incircle=True)->np.ndarray:
    """_summary_

    Args:
        mask (np.ndarray): binary mask image (1 channel)

    Returns:
        np.ndarray: shrun mask which is a little smaller
    """
    # Define a kernel (structuring element) for the following operation
    kernel = np.ones((3, 3), np.uint8)  # 5x5 kernel of ones
    # Perform the opening operation to remove noise
    opened_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if cut_incircle:
        opened_img = get_incircle_mask(opened_img)
    # Apply erosion to shrink the mask
    shrunk_mask = cv2.erode(opened_img, kernel, iterations=iterations)  
    return shrunk_mask

def get_incircle_mask(mask: np.ndarray) -> np.ndarray:
    """Get the incircle of a binary mask.

    Args:
        mask (np.ndarray): Binary mask image (1 channel).

    Returns:
        np.ndarray: Binary mask containing the largest inscribed circle.
    """
    # Find contours of the mask
    incircle_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(mask)  # Return an empty mask if no contours found
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, centre = cv2.minMaxLoc(distance)
    radius = int(max_val * 0.8)
    radius = 5 if radius < 10 else radius
    # Create a new mask for the incircle
    cv2.circle(incircle_mask, centre, radius, 1, thickness=-1)  # Fill the circle with 1s
    return incircle_mask

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

def combine_mask_list(mask_list)->np.ndarray:
    """Get the segmentation mask from SAM result.

    Args:
        mask_list (list): A list of masks (2D numpy arrays).

    Returns:
        np.ndarray: Combined mask.
    """
    result = reduce(lambda x, y: x + y, mask_list)
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

def mask_to_pc(mask:np.ndarray, depth_img, intrinsic, depth_scale=0.0010000000474974513)->np.ndarray:
    """_summary_

    Args:
        mask (np.ndarray): _description_
        depth_img (_type_): _description_
        depth_scale (float, optional): _description_. Defaults to 0.0010000000474974513.

    Returns:
        list: _description_
    """
    # Create a RealSense intrinsics object
    color_intrinsic = intrinsic

    seg_pc=[]
    # Convert the mask to a boolean array (True where mask == 1)
    mask_bool = mask == 1

    # Get the indices of the masked pixels (y, x)
    y_indices, x_indices = np.nonzero(mask_bool)

    # Get the depth values at the masked pixel locations and scale them
    depth_values = depth_img[y_indices, x_indices] * depth_scale

    # Filter out invalid depth values
    valid_depth_mask = (depth_values > 0) & (depth_values < 1.5)
    valid_y = y_indices[valid_depth_mask]
    valid_x = x_indices[valid_depth_mask]
    valid_depth_values = depth_values[valid_depth_mask]

    # Use rs.rs2_deproject_pixel_to_point to convert pixels to 3D points
    for x, y, depth in zip(valid_x, valid_y, valid_depth_values):
        result = rs.rs2_deproject_pixel_to_point(color_intrinsic, [x, y], depth)
        seg_pc.append(result)
    # Assuming voxel_down_pcd is your point cloud data that you want to downsample and clean
    # Perform statistical outlier removal on the downsampled point cloud
    pcd = o3d.geometry.PointCloud()  # Replace with your actual downsampled point cloud
    pcd.points = o3d.utility.Vector3dVector(seg_pc)  # Use your point data
    # # Apply voxel grid downsampling
    voxel_size = 0.001 # Adjust this value according to the density of your point cloud
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size)
    # # Remove statistical outliers
    # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
    return voxel_down_pcd

def generate_pc_pose(pc: np.ndarray):
    if pc.shape[0] <= 3:
        # Return dummy centroid and identity matrix as dummy pose
        dummy_centroid = np.array([0.0, 0.0, 0.0])
        dummy_pose = np.eye(3)  # Identity matrix as a dummy pose
        return dummy_centroid, dummy_pose
    
    # Standardize the point cloud
    std_scaler = StandardScaler()
    x_std = std_scaler.fit_transform(pc)  # Standardize in one step
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    pca.fit(x_std)  # Fit PCA to standardized data
    pose = pca.components_  # Extract PCA components
    
    # Calculate centroid point
    centroid_point = std_scaler.mean_
    
    return centroid_point, pose



# Function to find the nearest k points from multiple midpoints using SciPy's KDTree
def find_nearest_k_points(kdtree, mid_points, k=50):
    # Build the KDTree from the given points
    all_nearest_points = []

    # Iterate over each midpoint and perform KNN search
    for mid_point, valid_num in zip(mid_points, k):
        k = np.max([50, int(valid_num/5)])  # Limit k to a maximum of 500 or the number of valid points
        distances, idx = kdtree.query(mid_point, k=k, workers=6)  # Perform KNN search
        
        all_nearest_points.append(idx)  # Append the indices of nearest points

    return all_nearest_points

def create_kdtree(point_cloud):
    kdtree = KDTree(point_cloud.points)
    return kdtree

def find_valid_pixel_count(masks):
    # Initialize an empty list to hold counts of valid pixels
    valid_pixel_counts = []

    # Loop through each mask in the list
    for mask in masks:
        # Count non-zero pixels (valid pixels) in the current mask
        count = np.count_nonzero(mask)  # You can also use mask.sum() for binary masks
        valid_pixel_counts.append(count)
    
    return valid_pixel_counts