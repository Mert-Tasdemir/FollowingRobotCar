#!/bin/bash
cd /home/mert/source/FollowingRobotCar/
ping 127.0.0.1 -4  -w 1 -q
screen -dmS car python carMoveControlled2.py
# python carMoveControlled2.py

