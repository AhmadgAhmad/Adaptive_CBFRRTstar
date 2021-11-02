from matplotlib import pyplot
from shapely.geometry.point import Point
import shapely.affinity
from descartes import PolygonPatch
import itertools
from shapely.ops import cascaded_union
from rtree import index
# Note: download figures.py manually from shapely github repo, put it in shapely install directory
# from shapely.figures import SIZE, GREEN, GRAY, set_limits


# 1st elem = center point (x,y) coordinates
# 2nd elem = the two semi-axis values (along x, along y)
# 3rd elem = angle in degrees between x-axis of the Cartesian base
#            and the corresponding semi-axis
ellipse = ((0, 0),(7, 4),36)

# Let create a circle of radius 1 around center point:
circ = shapely.geometry.Point(ellipse[0]).buffer(1)
circ2 = shapely.geometry.Point((5,-5)).buffer(2)
circ

a = circ.i(circ2)


# Let create the ellipse along x and y:
ell  = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))


# Let rotate the ellipse (clockwise, x axis pointing right):
ellr = shapely.affinity.rotate(ell,ellipse[2])

# If one need to rotate it clockwise along an upward pointing x axis:
elrv = shapely.affinity.rotate(ell,90-ellipse[2])
# According to the man, a positive value means a anti-clockwise angle,
# and a negative one a clockwise angle.


fig = pyplot.figure()
ax = fig.add_subplot(111)
patch = PolygonPatch(elrv, fc='green', ec='gray', alpha=0.5, zorder=2)
patch2 = PolygonPatch(circ2, fc='red', ec='gray', alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.add_patch(patch2)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
pyplot.show()