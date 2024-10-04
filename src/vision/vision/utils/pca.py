import numpy as np
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

def angle_between_vectors(u, v):
    dot_product = sum(i*j for i, j in zip(u, v))
    norm_u = math.sqrt(sum(i**2 for i in u))
    norm_v = math.sqrt(sum(i**2 for i in v))
    cos_theta = dot_product / (norm_u * norm_v)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_rad, angle_deg