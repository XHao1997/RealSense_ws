import numpy as np
def generate_pc_pose(pc:np.ndarray):
    std_scaler = StandardScaler()
    std_scaler.fit(pc)
    x_std = std_scaler.transform(pc)
    pca = PCA(n_components=3)
    pca.fit_transform(x_std)
    
    centroid_point = std_scaler.mean_
    pose = pca.components_
    return centroid_point, pose

def angle_between_vectors(u, v):
    dot_product = sum(i*j for i, j in zip(u, v))
    norm_u = math.sqrt(sum(i**2 for i in u))
    norm_v = math.sqrt(sum(i**2 for i in v))
    cos_theta = dot_product / (norm_u * norm_v)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_rad, angle_deg