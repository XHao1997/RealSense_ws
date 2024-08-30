from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
def generate_launch_description():
    ld = LaunchDescription()
    teleop_key = ExecuteProcess(
            cmd=['xterm','-e','ros2', 'run', 'turtlesim', 'turtle_teleop_key'],
            name='teleop_key',
        )     
    data_collect = Node(
        package='data_collection',
        executable='collect_rgb',
        name='collect_rgb',
        output='log'
    )

    ld.add_action(teleop_key)       
    ld.add_action(data_collect)

    return ld