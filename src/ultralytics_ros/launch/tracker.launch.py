from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('yolo_model', default_value='best.pt'),
        DeclareLaunchArgument('input_topic', default_value='camera/camera/color/image_raw'),
        DeclareLaunchArgument('result_topic', default_value='/yolo_result'),
        DeclareLaunchArgument('result_image_topic', default_value='/yolo_image'),
        DeclareLaunchArgument('conf_thres', default_value='0.25'),
        DeclareLaunchArgument('iou_thres', default_value='0.45'),
        DeclareLaunchArgument('max_det', default_value='300'),
        DeclareLaunchArgument('tracker', default_value='bytetrack.yaml'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('result_conf', default_value='true'),
        DeclareLaunchArgument('result_line_width', default_value='1'),
        DeclareLaunchArgument('result_font_size', default_value='1'),
        DeclareLaunchArgument('result_font', default_value='Arial.ttf'),
        DeclareLaunchArgument('result_labels', default_value='true'),
        DeclareLaunchArgument('result_boxes', default_value='true'),

        # Set launch configurations
        SetLaunchConfiguration('use_sim_time', LaunchConfiguration('use_sim_time')),

        # Define the YOLO tracking node
        Node(
            package='ultralytics_ros',
            executable='tracker_node.py',
            output='screen',
            parameters=[{
                'yolo_model': LaunchConfiguration('yolo_model'),
                'input_topic': LaunchConfiguration('input_topic'),
                'result_topic': LaunchConfiguration('result_topic'),
                'result_image_topic': LaunchConfiguration('result_image_topic'),
                'conf_thres': LaunchConfiguration('conf_thres'),
                'iou_thres': LaunchConfiguration('iou_thres'),
                'max_det': LaunchConfiguration('max_det'),
                'tracker': LaunchConfiguration('tracker'),
                'result_conf': LaunchConfiguration('result_conf'),
                'result_line_width': LaunchConfiguration('result_line_width'),
                'result_font_size': LaunchConfiguration('result_font_size'),
                'result_font': LaunchConfiguration('result_font'),
                'result_labels': LaunchConfiguration('result_labels'),
                'result_boxes': LaunchConfiguration('result_boxes'),
                # Uncomment and modify the line below to specify classes
                # 'classes': [0, 1, 2],
                'device': LaunchConfiguration('device'),
            }]
        ),

        # Define the image view node (only if debug is true)
        Node(
            condition=IfCondition(LaunchConfiguration('debug')),
            package='image_view',
            executable='image_view',
            output='screen',
            remappings=[
                ('image', LaunchConfiguration('result_image_topic')),
            ]
        ),
    ])
