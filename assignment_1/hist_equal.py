__author__ = 'David'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MAX_INTENSITY = 255


def histogram(im):
    """
    Return the histogram of the image
    :param im: image read by cv2.imread
    :return: histogram
    """
    bin = np.zeros(MAX_INTENSITY)
    for intensity in im.ravel():
        # bin uses 0-based indexing
        bin[intensity - 1] += 1
    return bin


def cumulative(hist):
    """
    Transform original histogram to cumulative histogram
    :param hist: original histogram
    :return: cumul hist
    """
    cumul = np.copy(hist)
    for intensity in range(1, MAX_INTENSITY):
        cumul[intensity] += cumul[intensity - 1]
    return cumul


def equalized_intensity(cum_hist, orig_intensity):
    """
    Calculate the new pixel intensity after applying hist equal
    :param cum_hist:
    :param orig_intensity:
    :return:
    """
    max_cum = cum_hist[-1]
    pixel_cum = cum_hist[orig_intensity - 1]
    return pixel_cum / max_cum * MAX_INTENSITY


def equalize(im):
    """
    Generate the image transformed via histogram equalization
    :param im: original image
    :return: transformed image
    """
    equalized_im = np.zeros(im.size)
    cum_hist = cumulative(histogram(im))
    # Transform every pixel
    for index, intensity in enumerate(im.ravel()):
        equalized_im[index] = equalized_intensity(cum_hist, intensity)
    return equalized_im.reshape(im.shape)


def plot_figure(im, fig_index):
    x = np.arange(1, MAX_INTENSITY + 1)
    plt.figure(fig_index)

    # Plot the original image
    plt.subplot(221)
    plt.imshow(im, cmap=cm.Greys_r)
    plt.subplot(222)
    plt.plot(x, cumulative(histogram(im)))
    plt.xlim()
    plt.title('Original cumulative histogram')

    # Plot the equalized image
    equalized = equalize(im)
    plt.subplot(223)
    plt.imshow(equalized, cmap=cm.Greys_r)
    plt.subplot(224)
    plt.plot(x, cumulative(histogram(equalized)))
    plt.title('Equalized cumulative histogram')

if __name__ == '__main__':
    plot_figure(cv2.imread('haze.jpg', cv2.IMREAD_GRAYSCALE), 1)
    plot_figure(cv2.imread('airborne.jpg', cv2.IMREAD_GRAYSCALE), 2)
    plt.show()