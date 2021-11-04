import sys
print(sys.version)
from obstacle import Sphere, Ellipsoid, Wall
from params import Params

class World(object):
    def __init__(self,obstacles_list = []):
        self.params         = Params()
        self.width          = self.params.width
        self.length         = self.params.length
        self.obstacles_list = obstacles_list

    def add_obstacle(self):
