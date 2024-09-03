from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    ld = LaunchDescription()
    # load moveit config
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("realsense_collection"), "/launch", "/camera.launch.py"]
        ),
        launch_arguments={
            "use_sim_time": "true",
        }.items(),
    )

    teleop_key = ExecuteProcess(
            cmd=['xterm','-e','ros2', 'run', 'turtlesim', 'turtle_teleop_key'],
            name='teleop_key',
        )     
    data_collect = Node(
        package='realsense_collection',
        executable='collect_rgb',
        name='collect_rgb',
        output='screen'
    )
    ld.add_action(camera_launch)
    ld.add_action(teleop_key)       
    ld.add_action(data_collect)


    return ld