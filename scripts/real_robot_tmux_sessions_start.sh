#!/bin/bash

# session names
lcm_bridge="0-lcm_bridge"
med_planner="1-med_planner"
gripper_node="2-gripper_node"
bubbles="3-bubbles"
cameras="4-cameras"
pycharm_session_name='pycharm'
rviz_session_name='rviz'


# - LCM Bridge
tmux new-session -d -s $lcm_bridge
tmux send-keys -t $lcm_bridge 'roslaunch med_hardware_interface med_lcm_bridge.launch finger:=bubbles_flipped'

# - Med Planner
tmux new-session -d -s $med_planner
tmux send-keys -t $med_planner 'roslaunch arm_robots med.launch finger:=bubbles_flipped'

# - Gripper
tmux new-session -d -s $gripper_node
window=0
tmux rename-window -t $gripper_node:$window 'launch_gripper'
tmux send-keys -t $gripper_node:$window 'roslaunch wsg_50_driver wsg_50_tcp.launch'
window=1
tmux new-window -t $gripper_node:$window -n 'move_gripper'
tmux send-keys -t $gripper_node:$window 'rosservice call /wsg_50_driver/move 0 50'
tmux swap-window -s 1 -t 0

# - Bubbles
tmux new-session -d -s $bubbles
window=0
tmux rename-window -t $bubbles:$window 'launch_bubbles'
tmux send-keys -t $bubbles:$window 'roslaunch bubble_utils launch_bubbles.launch'
window=1
tmux new-window -t $bubbles:$window -n 'filter_bubbles'
tmux send-keys -t $bubbles:$window 'roslaunch bubble_utils filter_bubble_pointclouds.launch'
window=2
tmux new-window -t $bubbles:$window -n 'shear_estimation'
tmux send-keys -t $bubbles:$window 'roslaunch bubble_utils bubble_shear_estimation.launch'
tmux swap-window -s 2 -t 0


# - Cameras
tmux new-session -d -s $cameras
window=0
tmux rename-window -t $cameras:$window 'launch_cameras'
tmux send-keys -t $cameras:$window 'roslaunch mmint_camera_utils launch_cameras.launch'
window=1
tmux new-window -t $cameras:$window -n 'calibrate_cameras'
tmux send-keys -t $cameras:$window 'roslaunch mmint_camera_utils kuka_general_calibration.launch calibration_key:=drawing_front'
window=2
tmux new-window -t $cameras:$window -n 'tag_detections'
tmux send-keys -t $cameras:$window 'rosrun mmint_camera_utils camera_tag_detection.py --continuous --tag_ids 10 --rate 20 --time_on_buffer 1000'
window=3
tmux new-window -t $cameras:$window -n 'cartpole_rod_pose_estimation'
tmux send-keys -t $cameras:$window 'roslaunch bubble_control cartpole_tf_setup.launch'
window=4
tmux new-window -t $cameras:$window -n 'publish_camera_tfs'
tmux send-keys -t $cameras:$window 'rosrun mmint_camera_utils load_save_camera_tfs.py --load --filename tweezers_camera_sides --rate 1000'
tmux swap-window -s 4 -t 0

# - Pycharm
tmux new-session -d -s $pycharm_session_name
tmux send-keys -t $pycharm_session_name 'pycharm.sh'

# - Rviz
tmux new-session -d -s $rviz_session_name
tmux send-keys -t $rviz_session_name 'rosrun rviz rviz -d $(rospack find bubble_control)/rviz/cartpole.rviz'

# - Move robot
session_name='move_robot_mik'
tmux new-session -d -s $session_name
window=0
tmux rename-window -t $session_name:$window 'home_robot'
tmux send-keys -t $session_name:$window 'rosrun bubble_utils home_robot.py'
window=1
tmux new-window -t $session_name:$window -n 'set_grasp_pose'
tmux send-keys -t $session_name:$window 'rosrun bubble_utils set_grasp_pose.py'
tmux swap-window -s 1 -t 0


tmux attach-session -t $lcm_bridge
