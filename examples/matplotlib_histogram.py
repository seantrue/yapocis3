"""
Draw histogram of normal distribution using
"""
import numpy as np
import datetime
import matplotlib
import seaborn
import matplotlib.pyplot as plt
matplotlib.use("module://yapocis.lib.mplbackend")
seaborn.set_style("whitegrid")

if __name__ == '__main__':
    data = np.random.normal(0.0,1.0,10000)
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    plt.hist(data,bins=50)
    plt.title(f"{datetime.date.today()}: Normal distribution")
    plt.savefig("matplotlib_histogram.png")