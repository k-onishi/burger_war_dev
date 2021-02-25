#!/bin/bash

kill `ps auxww | grep JudgeWindow.py | grep -v grep| awk '{print $2}'`

gnome-terminal -- python ../catkin_ws/src/burger_war_kit/judge/JudgeWindow.py
sleep 1
bash ../catkin_ws/src/burger_war_kit/judge/test_scripts/init_single_play.sh ../catkin_ws/src/burger_war_kit/judge/marker_set/sim.csv localhost:5000 you enemy