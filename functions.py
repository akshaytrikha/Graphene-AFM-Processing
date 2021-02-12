# image processing functions used in Graphene-pipeline.ipynb
# Akshay Trikha
# 11th February, 2021

import cv2 as cv                            # OpenCV for image processing
import numpy as np                          # NumPy for quick maths
import matplotlib.pyplot as plt             # Matplotlib for visualizing
from numba import jit, njit, types, typeof  # optimization library
from numba.typed import Dict, List          # optimized data structures
# import time                                 # measure function execution times


# --- Image Processing Functions ---
def setup():
    """reads image performs set steps"""
    # get color and grayscale images
    color_image = cv.imread(image_name)
    gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

    
def normalize():
    """normalizes input so that microscope light is """
    pass


def clean_backround():
    """Implements Piper's algorithm to clean background"""
    pass


@njit
def get_areas_helper(watershed_markers, particle_areas):
    # loop through pixels in watershed markers and count the number of pixels for each color
    for row in range(1, len(watershed_markers) - 1):
        for col in range(1, len(watershed_markers[0]) - 1):
            # if pixel not in background
            if watershed_markers[row][col] != 1:
                # get current pixel and its neighbours 
                current = watershed_markers[row][col]
                # add current pixel to dictionary
                if current not in particle_areas:
                    particle_areas[current] = 1.0
                else:
                    particle_areas[current] += 1.0
    
    # remove -1 key from particle_areas because it represents contours drawn by cv.watershed()
    if -1 in particle_areas:
        del particle_areas[-1]
        
    # loop to adjust areas from number of pixels to nm^2
    for particle in particle_areas:
        current_area = particle_areas[particle] * nm_per_pixel**2
        particle_areas[particle] = current_area
                    
    return particle_areas


def get_areas(watershed_markers):
    """get the areas of the particles"""

    # dictionary mapping colors to their areas
    particle_areas = Dict.empty(
        key_type=types.int64, # don't need int64 but compiler throws warnings otherwise
        value_type=types.float64
    )

    particle_areas = get_areas_helper(watershed_markers, particle_areas)

    return particle_areas


# --- Visualization Functions ---
def display_images(images, titles, grayscales):
    """takes in list of images, list of titles, list of boolean grayscales and displays them"""
    if len(images) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(10,10))
        plt.imshow(images[0], cmap=plt.cm.gray)
        axs.set_title(titles[0])
    else:
        fig, axs = plt.subplots(1, len(images), figsize=(15,15))

        # loop through images and display
        for i in range(len(images)):
            if grayscales[i]:
                axs[i].imshow(images[i], cmap=plt.cm.gray)
                axs[i].set_title(titles[i])
            else:
                axs[i].imshow(images[i])
                axs[i].set_title(titles[i])
                
                
def save_images(images, titles):
    """saves images to disk"""
    for i in range(len(images)):
        plt.savefig(str(images[i]) + "_" + titles[i] + ".png", dpi=500)


def draw_point_contour(point, contour):
    """takes in a point and a contour and returns true if point is within contour and false otherwise"""
    # load new images
    color = cv.imread("./inputs/TES-36b-cropped.tif")
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    
    cv.drawContours(color, [contour], -1, (100, 255, 100), 2)
    color = cv.circle(color, point, 15, (255, 100, 100), 15)
    if cv.pointPolygonTest(contour, point, False) == 0.0:
        return True
    else:
        return False
    plt.imshow(color);

    
