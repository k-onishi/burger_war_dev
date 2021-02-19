#!/bin/bash

# sudo -E オプションをつける理由はproxy環境下でdocker proxy設定を継承するため。

HOME=/home/ubuntu

function install_ros_related_packages(){
    # joint state controller, and ros package
    sudo apt install -y ros-melodic-ros-control ros-melodic-ros-controllers  ros-melodic-joint-state-controller ros-melodic-effort-controllers ros-melodic-position-controllers ros-melodic-joint-trajectory-controller
    sudo apt install ros-melodic-cob-srvs
    # gazebo
    sudo apt-get install -y gazebo9
    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
    wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
    sudo apt-get update -y
    sudo apt-get install -y ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
    echo "export GAZEBO_MODEL_PATH=:${HOME}/catkin_ws/src/ai_race/ai_race:${HOME}/catkin_ws/src/ai_race/ai_race/sim_world/models" >> ~/.bashrc
    export GAZEBO_MODEL_PATH=:${HOME}/catkin_ws/src/ai_race/ai_race:${HOME}/catkin_ws/src/ai_race/ai_race/sim_world/models
    # camera image
    sudo apt-get install -y ros-melodic-uvc-camera
    sudo apt-get install -y ros-melodic-image-*
}

function install_torch(){
    ### pytorch from pip image (v1.4)
    sudo apt-get install -y libopenblas-base libopenmpi-dev
    sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    
    python -m pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp27-cp27mu-linux_x86_64.whl
    python -m pip install torchvision==0.2.2
    pip3 install torch==1.4.0 torchvision==0.2.2
    pip install 'pillow<7'
}

function install_sklearn(){
    ### sklearn python3
    pip3 install scikit-learn
    # pip3 install matplotlib
    #sudo apt-get -y install python3-tk
}

function install_numpy(){
    echo "skip"
    ### pandas python2,3 (defaultを使えばよい)
    #pip3 install cython
    #pip3 install numpy
    pip3 install -U pandas
}

function install_opencv(){
    pip3 install opencv-python==3.4.10.37
    ### opencv python
    ### opencv python はソースからビルドする必要がある. 8～10時間ほど掛かる.
    # cd ~
    # sudo rm -rf nano_build_opencv
    # git clone https://github.com/mdegans/nano_build_opencv
    # cd nano_build_opencv
    # yes | ./build_opencv.sh 3.4.10
}

install_ros_related_packages
install_torch
install_sklearn
install_numpy
install_opencv
