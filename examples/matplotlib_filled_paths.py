"""
Example of drawing matplot areas using kerneldraw
"""
import matplotlib
import numpy as np
import datetime

matplotlib.use("module://yapocis.lib.mplbackend")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

verts = [
   (0., 0.),  # left, bottom
   (0., 1.),  # left, top
   (1., 1.),  # right, top
   (1., 0.),  # right, bottom
   (0., 0.),  # ignored
]
diamond_verts = [
    (0,-1),
    (1,0),
    (0,1),
    (-1,0),
    (0,-1),
]
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]


if __name__ == '__main__':

    path = Path(verts, codes)
    dpath = Path(diamond_verts, codes)


    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = fig.add_subplot(111,rasterized=True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # Draw cross hairs
    plt.axvline(0,color="yellow")
    plt.axhline(0,color="grey")
    # Locate text lower left and upper right
    plt.text(-1.1,-1.1, "lb")
    plt.text(1.1,1.1, "rt")
    # Square orange patch to UR quadrang
    patch = patches.PathPatch(path, facecolor='orange')
    ax.add_patch(patch)
    # Diamond pink patch centered
    dpatch = patches.PathPatch(dpath, facecolor='pink',alpha=.5)
    ax.add_patch(dpatch)
    # Green circle upper left quadrant
    circle = Path.circle((-1,1),radius=.5)
    cpatch = patches.PathPatch(circle, facecolor='green',alpha=.5)
    ax.add_patch(cpatch)
    # Filled in sine wave
    x = np.arange(-1,1,.01)
    xr = x * np.pi*5
    y = np.sin(xr)*.5
    y -= .5
    x[0] = x[1]
    y[0] = -1.5
    y[-1] = -1.5
    x[-1] = x[-2]
    scodes = [Path.MOVETO]
    scodes += [Path.LINETO]*(len(y)-1)
    scodes += [Path.CLOSEPOLY]
    vertices = np.vstack((x,y)).transpose()
    vertices = np.concatenate((vertices,[[0,0]]))
    spath = Path(vertices, scodes)
    plt.plot(x,y)
    spatch = patches.PathPatch(spath,facecolor='blue',alpha=.5)
    ax.add_patch(spatch)
    plt.title(str(datetime.date.today()))
    plt.savefig("matplot_filled_paths.png")