import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Enable both depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start the pipeline with the configuration
pipeline_profile = pipeline.start(config)

# Create an align object for aligning depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Get the active profile and streams
profile = pipeline.get_active_profile()

# Get depth stream profile and intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
print("Depth Intrinsics:", depth_intrinsics)

# Get color stream profile and intrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()
print("Color Intrinsics:", color_intrinsics)

# Align frames and get aligned frames
for i in range(5):  # Skip a few frames for initialization
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

# Get aligned depth and color frames
aligned_depth_frame = aligned_frames.get_depth_frame()
aligned_color_frame = aligned_frames.get_color_frame()

if not aligned_depth_frame or not aligned_color_frame:
    print("Could not get aligned frames")
else:
    print("Aligned depth and color frames obtained")

# Stop the pipeline
pipeline.stop()
