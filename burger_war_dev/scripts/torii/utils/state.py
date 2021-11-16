#!python3
#-*- coding: utf-8 -*-

from collections import namedtuple

State = namedtuple (
    'State', ('pose', 'lidar', 'image', 'mask')
)