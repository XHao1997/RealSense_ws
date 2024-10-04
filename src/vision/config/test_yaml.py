import yaml

# Load intrinsics from the YAML file
with open('camera_intrinsics.yaml', 'r') as yaml_file:
    intrinsics = yaml.safe_load(yaml_file)

# Access the values
color_intrinsic = intrinsics['camera_intrinsics']
print(color_intrinsic)