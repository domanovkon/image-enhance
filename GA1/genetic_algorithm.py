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
# Соритровка популяции по значениям фитнес-функции
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def population_sort(images_population):
    return images_population[images_population[:, 5].argsort()]


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def gen_alg(image, image_bordered):
    epochs = 30
    populationSize = 100

    global_brightness_value = global_brightness_value_calc(image)
    images_population = generate_population(populationSize)

    for i in range(epochs):
        for i in range(populationSize):
            images_population[i, 5] = chromosome_improve(images_population[i], image, image_bordered,
                                                         global_brightness_value)

    sorted_population = population_sort(images_population)

    a = 5
