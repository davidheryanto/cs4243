import matplotlib.pyplot as plt
import numpy as np
from skimage import data


def show_image(image_path):
    """
    http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive
    :param image_path:
    :return:
    """
    """
    :param image_path:
    :return:
    """
    plt.imshow(data.imread(image_path))
    plt.pause(0.1)

if __name__ == '__main__':
    x = [1, 2, 3]
    plt.ion()  # turn on interactive mode
    show_image('input/london_bridge.jpg')
    plt.pause(2.0)
    show_image('input/california.png')
    plt.pause(5.0)
