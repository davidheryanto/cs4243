__author__ = 'David'

from skimage import data, img_as_float, img_as_int
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def method1(image):
    """
    :param image:
    :return:
    """
    n_intensity_level = 255
    rank = image.ravel().argsort().argsort()
    pixels_per_intensity_level = image.size / n_intensity_level
    return (rank / pixels_per_intensity_level).reshape(image.shape)


def method2(image):
    """
    Adapted from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    :param image:
    :return:
    """
    frequency = np.zeros(255, dtype=int)
    for intensity_level in image.ravel():
        frequency[intensity_level] += 1

    cumulative_frequency = np.zeros(255, dtype=int)
    for i, level in enumerate(frequency):
        cumulative_frequency[i] = level + cumulative_frequency[i - 1] if i > 0 else level

    # Use linear interpolation to get new intensity level for each original intensity level.
    # The cumulative frequency will be the basis for the interpolation.
    return (np.interp(image.flat, np.arange(255), cumulative_frequency)).reshape(image.shape)


if __name__ == '__main__':
    image = data.imread('input/london_bridge.jpg')

    if image.ndim > 2:
        print('Currently only support grayscale image')
        exit(0)

    plt.subplot(131)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original Image')

    image1 = method1(image)
    plt.subplot(132)
    plt.imshow(image1, cmap=plt.cm.gray)
    plt.title('Method 1')

    image2 = method2(image)
    plt.subplot(133)
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.title('Method 2')

    # Maximize window and show
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()