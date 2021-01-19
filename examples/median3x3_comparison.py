import numpy as np
import time
from scipy.ndimage.filters import median_filter
from yapocis.kernels import median


def median3x3(input, iterations=1):
    for i in range(iterations):
        input = median_filter(input,3)
    return input

sizes = []
areas = []
yapocisfast_deltas = []
yapocis_deltas = []
numpy_deltas = []
errors = []
ITERATIONS=20
SAMPLES=50
STEP=40
for i in range(0,SAMPLES):
    width = (i+1)*STEP
    img = np.float32(np.random.random_sample((width,width)))
    
    t = time.time()
    output1 = median3x3(img,ITERATIONS)
    numpy_delta = time.time() - t
    
    t = time.time()
    output2 = median.median3x3slow(img, ITERATIONS)
    yapocis_delta = time.time() - t
    
    t = time.time()
    output3 = median.median3x3fast(img, ITERATIONS)
    yapocisfast_delta = time.time() - t

    print(width, numpy_delta, yapocis_delta, yapocisfast_delta)
    sizes.append(width)
    areas.append(width*width)
    numpy_deltas.append(numpy_delta)
    yapocis_deltas.append(yapocis_delta)
    yapocisfast_deltas.append(yapocisfast_delta)
    
from matplotlib import pyplot as plot
areas = np.array(areas)
numpy_deltas = np.array(numpy_deltas)
yapocis_deltas = np.array(yapocis_deltas)
yapocisfast_deltas = np.array(yapocisfast_deltas)
ratio = yapocis_deltas/numpy_deltas
ratio2 = yapocisfast_deltas/numpy_deltas
p = plot.subplot(111)
plot.title("Median 3x3 filter, %s iterations" % ITERATIONS,fontsize="large")
l1 = plot.plot(areas, numpy_deltas,color="lightgreen")
l2 = plot.plot(areas, yapocis_deltas,color="pink")
l3 = plot.plot(areas, yapocisfast_deltas,color="yellow")
l4 = p.plot(areas,ratio,color="red")
l5 = p.plot(areas,ratio2,color="black")
plot.fill_between(areas, numpy_deltas, yapocis_deltas,color="lightgreen")
plot.fill_between(areas, yapocisfast_deltas, yapocis_deltas, color="pink")
plot.fill_between(areas, np.zeros_like(yapocis_deltas),yapocisfast_deltas, 
                  color="yellow")
plot.figlegend( (l1, l2, l3, l4, l5), 
                ('numpy', 'yapocis','yapocis-fast','yapocis/numpy','fast/numpy'), 
                'upper left', shadow=True)
plot.xlabel("Pixels",fontsize="medium")
plot.ylabel("Seconds",fontsize="medium")
plot.savefig("../../doc/images/median3x3-performance.png")
plot.show()
