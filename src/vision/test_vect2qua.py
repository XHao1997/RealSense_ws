import numpy as np
from scipy.spatial.transform import Rotation as R

# Assume you have 3 vectors representing the axes (e.g., x, y, z) of the coordinate frame
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, -1, 0])
z_axis = np.array([0, 0, -1])

# Construct the rotation matrix from the axes
rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

# Convert the rotation matrix to a quaternion
rotation = R.from_matrix(rotation_matrix)
quaternion = rotation.as_quat()  # [x, y, z, w]

print("Quaternion: ", quaternion)
