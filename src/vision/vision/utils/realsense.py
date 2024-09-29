import pyrealsense2 as rs
import numpy as np

def reset_hardware():
    """reset camera
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

def Initialize_stream():
    """Initialize the pipeline for streaming

    Returns:
        a pipline of realsense camera capturing images.
        and profile store all configuration
    """

    # Initialize the pipeline for streaming
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile, pipeline.start(config)
    return pipeline, profile

def align_depth_to_color(pipeline):
    """align depth frame to color frame

    Args:
        pipeline (rs2.pipeline): pipeline ready for streaming

    Returns:
        aligned_frames: user can get aligned depth and color frame
    """
    # aligner
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Wait for frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    return aligned_frames

def get_desired_frame_from_aligned_frame(key:str):
    """get desired frame(color/depth)

    Args:
        key (str): the name of the sensor frame you want to choose

    Returns:
        realsense_frame: the streaming frame user chooses
    """
    match key:
        case 'color':
            return aligned_frames.get_color_frame()
        case 'rgb':
            return aligned_frames.get_color_frame()
        case 'depth':
            return aligned_frames.get_depth_frame()

def get_instrinsics_from_frame(frame):
    """get intrinsics from realsense frame

    Args:
        frame (realsense frame): The frame store depth or color image

    Returns:
        _type_: the intrinsics of the 
    """
    return frame.profile.as_video_stream_profile().intrinsics

def deproject_pixel_to_point(color_intrinsics, 
                             dist_value:float, 
                             pixel_pos:list,
                             ):
    """convert one point in the depth pixel to the world coordinate

    Args:
        color_intrinsics (pyrealsense): the intrinsic of the rgb frame
        dist_value (float): the distance value get from scaling the raw 
        depth value
        pixel_pos (list): the position of the desired point in rgb image

    Returns:
        list: [x,y,z] of the point in pixel
    """
    if treshold[0]<dist_value<threshold[1]:
        point_pos = rs.rs2_deproject_pixel_to_point(color_intrinsics, 
        [x, y], dist_value)

    return point_pos

def convert_depth_to_dist(depth_frame, position:list):
    """convert raw depth value to distance

    Args:
        depth_frame (realsense_frame): the chosen frame(depth)
        position (list): the pixel position for calculating distance

    Returns:
        float: the converted distance value in meter
    """
    x, y = position
    dist_value = depth_frame.get_distance(x, y)
    return dist_value                                                                                                                                                  