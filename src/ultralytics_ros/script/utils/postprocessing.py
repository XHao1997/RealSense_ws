import copy
import numpy as np
import cv2
from functools import reduce
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