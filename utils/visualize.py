from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib


def visualize_field(field, magnitude):
    X, Y = np.meshgrid(range(field.shape[1]), range(field.shape[2]), indexing='ij')

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    U = field[0, :, :]
    V = field[1, :, :]

    # for x in range(mpm.shape[1]):
    #     for y in range(mpm.shape[0]):
    plt.quiver(X, Y, U, V, magnitude, cmap='winter')
    plt.colorbar(cmap='winter')
    plt.show()
    pass

if '__name__' == "__main__":
    magnitude = np.zeros((64, 64))
    mpm = np.zeros((64, 64, 3))
    magnitude[32, 32] = 255
    magnitude = gaussian_filter(magnitude, sigma=6)
    magnitude = magnitude/magnitude.max()
    
    # for x in range(64):
    #     for y in range(64):
    #         vec = np.array([42, 42]) - np.array([y, x])
    #         vec = np.append(vec, 5)
    #         vec = vec / np.linalg.norm(vec) * magnitude[y, x] * 2
    #         mpm[y, x] = vec
    
    # visualize_mpm(mpm[15:-15, 15:-15], magnitude[15:-15, 15:-15])
    
    cord = np.array(np.meshgrid(range(64), range(64), indexing='ij'))
    field = np.array([32, 32]).reshape([2, 1, 1]) - cord
    field = field / np.maximum(np.linalg.norm(field, axis=0, keepdims=True), 1) * magnitude.reshape((1, 64, 64))
    visualize_field(field[:, 15:-15, 15:-15], magnitude[15:-15, 15:-15])
