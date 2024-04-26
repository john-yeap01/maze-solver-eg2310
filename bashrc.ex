alias cw='cd ~/colcon_ws'
alias cs='cd ~/colcon_ws/src'
alias ca='cd ~/colcon_ws/src/auto_nav/auto_nav'

alias cban='cd ~/colcon_ws && colcon build --packages-select auto_nav'
alias si='source install/setup.bash'
alias cb='cd ~/colcon_ws && colcon build --symlink-install && source ~/.bashrc'
source /opt/ros/foxy/setup.bash
source ~/colcon_ws/install/local_setup.bash

alias cw='cd ~/colcon_ws'
alias cs='cd ~/colcon_ws/src'
alias cb='cd ~/colcon_ws && colcon build --symlink-install && source ~/.bashrc'
source /opt/ros/foxy/setup.bash
source ~/colcon_ws/install/local_setup.bash
export ROS_DOMAIN_ID=30 #TURTLEBOT3
export ROS_DOMAIN_ID=38 #TURTLEBOT3

export TURTLEBOT3_MODEL=burger
alias sshrp='ssh ubuntu@`ssh aws cat rpi.txt`'
alias sshrp2='ssh ubuntu@`ssh aws cat rpi2.txt`'
alias rteleop='ros2 run turtlebot3_teleop teleop_keyboard'
alias rslam='ros2 launch turtlebot3_cartographer cartographer.launch.py'

export GAZEBO_MODEL_PATH=~/colcon_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
