import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_radius(ax, origin, radius):
    '''
        From StackOverflow question:
        https://stackoverflow.com/questions/13685386/
    '''
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax, zoom=1.):
    '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect("equal") and ax.axis("equal") not working for 3D.
        input:
          ax:   a matplotlib axis, e.g., as output from plt.gca().

    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) / zoom
    set_axes_radius(ax, origin, radius)





points = [[3.2342, 1.8487, -1.8186],
           [2.9829, 1.6434, -1.8019],
           [3.4247, 1.5550, -1.8093]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0] #first vector
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0] #sec vector

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx] #cross product

point  = np.array(p1)
normal = np.array(u_cross_v)

d = -point.dot(normal)

print('plane equation:\n{:1.4f}x + {:1.4f}y + {:1.4f}z + {:1.4f} = 0'.format(normal[0], normal[1], normal[2], d))

xx, yy = np.meshgrid(range(5), range(5))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
widths = np.linspace(2, 5, 20)
# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.quiver(x0, y0, z0, normal[0], normal[1], normal[2], color="red")

plt3d.plot_surface(xx, yy, z)

plt3d.set_xlabel("X", color='red', size=18)
plt3d.set_ylabel("Y", color='green', size=18)
plt3d.set_zlabel("Z", color='b', size=18)

# insert these lines
ax = plt.gca()
set_axes_equal(ax)
plt.show()