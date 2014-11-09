import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_rgb(filepath):
    bgr = cv2.imread(filepath)
    b,g,r = cv2.split(bgr)
    return cv2.merge([r,g,b])

def save_surfaces(surfaces, output_filepath):
    import pickle
    with open(output_filepath, 'wb') as file:
        pickle.dump(surfaces, file)

def load_surfaces(input_filepath):
    import pickle
    with open(input_filepath, 'rb') as file:
        return pickle.load(file)