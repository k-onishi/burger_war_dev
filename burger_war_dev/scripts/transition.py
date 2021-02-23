#!python3
#-*- coding: utf-8 -*-

from collections import namedtuple
from state import State

State = namedtuple (
    'State', ('lidar', 'map', 'image')
)
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)