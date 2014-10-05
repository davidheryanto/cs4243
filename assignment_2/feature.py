__author__ = 'David'

import operator
import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.linalg as linalg

# TODO
# from joblib import Parallel

# Progress indicator for eigenvalue calculation
progress = [x / 100.0 for x in range(0, 100, 5)]


def a(b,c):
    """

    :param b:
    :param c:
    :return:
    """

def gradient(window1, window2, distance):
    """
    Get the gradient between two image windows (patches)
    :param window1:
    :param window2:
    :param distance: the distance b/w 2 windows
    :return: gradient value
    """
    intensity1 = np.sum(window1.ravel()) / window1.shape[0]
    intensity2 = np.sum(window2.ravel()) / window2.shape[0]
    return (intensity2 - intensity1) / distance


def matrix(start_x, start_y, window_size, im):
    """
    Return the matrix for eigenvalue calculation
    :param start_x:
    :param start_y:
    :param window_size:
    :param im: image source
    :return:
    """
    im = im.astype('uint32')  # To avoid overflow
    mat = np.zeros((2, 2))

    # Check for boundaries
    height, width = im.shape[:2]
    if start_x + window_size > width or start_y + window_size > height:
        raise ValueError('Start position + window size exceeds boundary')

    start_x_right = start_x + window_size / 2
    start_y_bottom = start_y + window_size / 2

    window_current = im[start_y:start_y + window_size, start_x:start_x + window_size]
    window_right = im[start_y:start_y + window_size, start_x_right:start_x_right + window_size]
    window_bottom = im[start_y_bottom:start_y_bottom + window_size, start_x: start_x + window_size]

    gradient_x = gradient(window_current, window_right, window_size / 2)
    gradient_y = gradient(window_current, window_bottom, window_size / 2)

    mat[0][0] = gradient_x * gradient_x
    mat[0][1] = mat[1][0] = gradient_x * gradient_y
    mat[1][1] = gradient_y * gradient_y

    return mat


def eigenvalues(im, window_size):
    """
    Calculate the eigenvalues for the image
    :param im:
    :param window_size:
    :return: dictionary of eigenvalues eig; eig[(x,y)]= min eigenvalue for window with (x,y) as top-left pt
    """
    eig = {}
    height, width = im.shape[:2]

    for x in range(0, width - window_size, window_size / 2):
        for y in range(0, height - window_size, window_size / 2):
            print_progress(float(x) / width, float(y) / height)
            values = linalg.eig(matrix(x, y, window_size, im))[0]
            eig[(int(x), int(y))] = np.min(values)

    return eig


def print_progress(x_frac, y_frac):
    global progress
    for frac in progress:
        if x_frac > frac and y_frac > frac:
            print('{} % Done'.format(frac * 100))
            progress.remove(frac)
            return


if __name__ == '__main__':
    start = time.time()

    im = cv2.imread('lab_photo.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    eig = eigenvalues(im, window_size=20)
    eig_sorted = sorted(eig.items(), key=operator.itemgetter(1), reverse=True)

    for (x, y), val in eig_sorted[:100]:
        plt.scatter(x, y, s=50, facecolors='none', edgecolors='r')


    end = time.time()
    print('Execution time: {} sec'.format(end - start))

    plt.imshow(im, cmap=cm.Greys_r)
    plt.show()
