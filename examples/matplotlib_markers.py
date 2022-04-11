import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("module://yapocis.lib.mplbackend")

# Fixing random state for reproducibility
if __name__ == '__main__':

    np.random.seed(19680801)

    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.sqrt(x**2 + y**2)

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.set_size_inches(10,10)

    # marker symbol
    axs[0, 0].scatter(x, y, s=200, c=z, marker="o")
    axs[0, 0].set_title("marker='o'")

    # marker from TeX
    axs[0, 1].scatter(x, y, s=200, c=z, marker=r'$\alpha$')
    axs[0, 1].set_title(r"marker=r'\$\alpha\$'")

    # marker from path
    verts = [[-1, -1], [1, -1], [1, 1], [-1, -1]]
    axs[0, 2].scatter(x, y, s=200, c=z, marker=verts)
    axs[0, 2].set_title("marker=verts")

    # regular polygon marker
    axs[1, 0].scatter(x, y, s=200, c=z, marker=(5, 0))
    axs[1, 0].set_title("marker=(5, 0)")

    # regular star marker
    axs[1, 1].scatter(x, y, s=200, c=z, marker=(5, 1))
    axs[1, 1].set_title("marker=(5, 1)")

    # regular asterisk marker
    axs[1, 2].scatter(x, y, s=200, c=z, marker=(5, 2))
    axs[1, 2].set_title("marker=(5, 2)")

    plt.tight_layout()
    plt.show()
    plt.savefig("matplotlib_markers.png")
