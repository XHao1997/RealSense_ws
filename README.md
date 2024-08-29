- mkdir realsense_ws
- git clone git@github.com:XHao1997/RealSense_ws.git
- colcon buld && source ./install/setup.sh
- to run record data node, run these to commands in two terminals:
    1. ros2 run data_collection collect_rgb
    2. ros2 run data_collection telekey
