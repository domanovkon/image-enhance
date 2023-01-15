import cv2
import numpy as np
import math
import random

from transformation_kernel import *
from main import *
from transformation_kernel import *


# -----------------------------------------------------------
# Генерация начальной популяции
#   a ∈ [0 , 1.5]
#   b ∈ [0 , 0.5]
#   c ∈ [0 , 1]
#   k ∈ [0.5 , 1.5]
# -----------------------------------------------------------
def generate_population(population_size):
    n_values = [3, 5, 7]

    generatedMatrix = np.zeros((population_size, 6))
    generatedMatrix[:, 0] = np.round(np.random.uniform(0, 1.5, population_size), 2)
    generatedMatrix[:, 1] = np.round(np.random.uniform(0, 0.5, population_size), 1)
    generatedMatrix[:, 2] = np.round(np.random.uniform(0, 1, population_size), 2)
    generatedMatrix[:, 3] = np.round(np.random.uniform(0.5, 1.5, population_size), 2)
    generatedMatrix[:, 4] = np.random.choice(n_values, population_size)

    return generatedMatrix


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def gen_alg(image, image_bordered):
    epochs = 1
    populationSize = 100

    global_brightness_value = global_brightness_value_calc(image)
    images_population = generate_population(populationSize)

    for i in range(populationSize):
        images_population[i, 5] = transformaton_calculation(image, image_bordered, int(images_population[i][4]),
                                                            int(images_population[i][4] // 2), global_brightness_value)
    a = 5

