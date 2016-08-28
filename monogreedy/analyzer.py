from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def draw2dsurface(X, Y, zf):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(X, Y)
    Z = X*0
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j] = zf([X[i][j], Y[i][j]])

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(np.min(Z.flatten()), np.max(Z.flatten()))

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
