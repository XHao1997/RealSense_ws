import utils.postprocessing
import numpy as np
import cv2
import open3d as o3d
depth_img = cv2.imread("/home/wx_pc/Desktop/RealSense_ws/depth.png", cv2.IMREAD_UNCHANGED)
mask = np.load("/home/wx_pc/Desktop/RealSense_ws/mask.npy")[:,:,0]

print(mask.shape)
pcd = utils.postprocessing.mask_to_pc(mask, depth_img)
print(mask)
# utils.postprocessing.visualise_pc(pcd, is_numpy=False)
utils.postprocessing.generate_pc_pose(np.asarray(pcd.points))
# Generate the pose (centroid and orientation) from the point cloud
centroid, pose = utils.postprocessing.generate_pc_pose(np.asarray(pcd.points))

# Display the centroid and principal components in the Open3D viewer
# Create a sphere to represent the centroid
centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
centroid_sphere.translate(centroid)
centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Color the centroid red

# Create lines to represent the principal axes (X, Y, Z) from the pose
axis_length = 0.1  # Length of the principal axis lines

# X-axis (first principal component)
x_axis = o3d.geometry.LineSet()
x_axis.points = o3d.utility.Vector3dVector([centroid, centroid + pose[0] * axis_length])
x_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
x_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red for X-axis

# Y-axis (second principal component)
y_axis = o3d.geometry.LineSet()
y_axis.points = o3d.utility.Vector3dVector([centroid, centroid + pose[1] * axis_length])
y_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
y_axis.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green for Y-axis

# Z-axis (third principal component)
z_axis = o3d.geometry.LineSet()
z_axis.points = o3d.utility.Vector3dVector([centroid, centroid + pose[2] * axis_length])
z_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
z_axis.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue for Z-axis

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the geometries (point cloud, centroid, and principal axes)
vis.add_geometry(pcd)
vis.add_geometry(centroid_sphere)
vis.add_geometry(x_axis)
vis.add_geometry(y_axis)
vis.add_geometry(z_axis)

# Access visualization options and set the line width
opt = vis.get_render_option()
opt.line_width = 100.0  # Set the line width (default is usually 1.0)

# Run the visualizer
vis.run()
vis.destroy_window()