from collections import namedtuple

State = namedtuple (
    'State', ('pose', 'lidar', 'image', 'mask')
)
