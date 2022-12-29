import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
from random import randint
from numpy.random import default_rng
import random
# import time
from tqdm.notebook import tqdm_notebook

rng = default_rng()
np.random.seed(909)

from ench import *


def count_edges(img_array):
    edges = cv2.Canny(np.uint8(img_array), threshold1=100, threshold2=200)
    edge_count = np.count_nonzero(edges)
    return edge_count


def fitness_function(population, gray_levels, img_array, freq):
    fitness = []
    for chromosome in population:
        fitness.append(fitness_func_for_one(chromosome, gray_levels, img_array, freq))
    return np.array(fitness)


def fitness_func_for_one(chromosome, gray_levels, image_array, freq):
    enhanced_img_array = create_enhanced_image(gray_levels, chromosome, image_array)
    edge_count = count_edges(enhanced_img_array)

    intensities = sum([chromosome[i] * freq[i] for i in range(len(chromosome))])
    return np.log(np.log(intensities)) * edge_count


def evaluation(img, best_contrasted_img, equalized):
    return count_edges(best_contrasted_img), count_edges(equalized), count_edges(img)


def image_comparison(image, hist_equalized_img, improved_image):
    initial_edge_count = count_edges(image)
    hist_edge_count = count_edges(hist_equalized_img)
    improved_edge_count = count_edges(improved_image)
    print("Количество ребер исходного изображения: ", initial_edge_count)
    print("Количество ребер после эквализации гистограммы: ", hist_edge_count)
    print("Количество ребер улучшенного изображения: ", improved_edge_count)
    return initial_edge_count, hist_edge_count, improved_edge_count
