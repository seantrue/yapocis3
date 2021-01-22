import numpy as np
from yapocis.rpc import kernels
draw_cl = None
kernels.import_program("draw", draw="draw_cl")

def draw(lines, image, color=0.0):
    lines = np.array(lines)
    lines = lines.flatten()
    image = draw_cl(lines, image, color, global_size=len(lines)//4)
    return image

def test_draw():
    from yapocis.utils import show_array, stage
    from random import randint
    a = np.ones((1000,1000)).astype(np.float32)
    def rx():
        return randint(0,a.shape[0])
    def ry():
        return randint(0,a.shape[1])
    for nlines in range(128,4092,127):
        a[:,:] = 1.0
        lines = []
        length = 40
        for iline in range(nlines):
            x,y = rx(), ry()
            x1,y1 = x+randint(-length,length),y+randint(-length,length)
            x = max(0,min(999,x))
            y = max(0,min(999,y))
            x1 = max(0,min(999,x1))
            y1 = max(0,min(999,y1))
            lines.append([x,y,x1,y1])
        stage("draw %s" % len(lines))
        draw(lines, a)
        stage()
        show_array("draw %s" % nlines, a)
        stage("drawf %s" % len(lines))

if __name__ == "__main__":
    test_draw()      
