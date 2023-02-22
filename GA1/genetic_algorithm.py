import cv2
import numpy as np
import math
import random

from utils import *
from transformation_kernel import *
from main import *


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
# Функция отбора особей на основе бинарного турнира
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def binary_tournament(population, sample_size):
    selected_parents = []
    population_size = len(population)
    for i in range(sample_size):
        i = random.randint(0, population_size - 1)
        j = random.randint(0, population_size - 1)
        while i == j:
            j = random.randint(0, population_size - 1)
        if population[i, 5] >= population[j, 5]:
            selected_parents.append(population[i])
        else:
            selected_parents.append(population[j])
    return selected_parents


# -----------------------------------------------------------
# Функция арифметического кроссовера
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def crossover(parents):
    childs = []
    parents_size = len(parents) // 2
    for i in range(parents_size):
        parent1_index = random.randint(0, parents_size - 1)
        parent2_index = random.randint(0, parents_size - 1)
        while parent1_index == parent2_index:
            parent2_index = random.randint(0, parents_size - 1)
        child1 = np.zeros(6)
        child2 = np.zeros(6)
        random_value = random.uniform(0, 1)
        for j in range(4):
            child1[j] = np.round(
                random_value * parents[parent1_index, j] + (1 - random_value) * parents[parent2_index, j], 2)
            child2[j] = np.round(
                random_value * parents[parent2_index, j] + (1 - random_value) * parents[parent1_index, j], 2)
        if random_value >= 0.5:
            child1[4] = parents[parent1_index, 4]
            child2[4] = parents[parent2_index, 4]
        else:
            child1[4] = parents[parent2_index, 4]
            child2[4] = parents[parent1_index, 4]
        child1[1] = np.round(child1[1], 1)
        child2[1] = np.round(child2[1], 1)
        childs.append(child1)
        childs.append(child2)
    return childs


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def gen_alg(image):
    epochs = 20
    populationSize = 100
    k = int(np.round(populationSize * 0.1))

    global_brightness_value = global_brightness_value_calc(image)

    images_population = generate_population(populationSize)

    for i in range(populationSize):
        images_population[i, 5] = chromosome_improve(images_population[i], image,
                                                     global_brightness_value)

    fitness_values_array = []

    for i in range(epochs):
        sorted_population = population_sort(images_population)
        k_best = sorted_population[-k:]

        fitness_values_array.append(k_best[-1, 5])

        selected_parents = binary_tournament(sorted_population[:populationSize - k], int(populationSize / 2 - k))

        parents = np.concatenate((k_best, selected_parents), axis=0)
        children = np.stack(crossover(parents), axis=0)

        for i in range(len(children)):
            children[i, 5] = chromosome_improve(children[i], image,
                                                global_brightness_value)
        images_population = np.concatenate((parents, children), axis=0)

    final_population = population_sort(images_population)
    best_chromo = final_population[-1]

    plot_generations_graph(fitness_values_array)

    n = int(best_chromo[4])
    off = n // 2
    new_image_bordered = make_mirror_reflection(image, off)

    # n = best_chromo[4]
    # off = n // 2
    return transformaton_calculation(image, new_image_bordered, n, off, global_brightness_value, best_chromo)
