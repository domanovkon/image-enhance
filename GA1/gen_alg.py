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
def binary_tournament(population):
    selected_parents = []
    chromosomes = population.copy()
    np.random.shuffle(chromosomes)
    for i in range(0, len(chromosomes), 2):
        if (chromosomes[i, 5] >= chromosomes[i + 1, 5]):
            selected_parents.append(chromosomes[i])
        else:
            selected_parents.append(chromosomes[i + 1])
    return selected_parents


# -----------------------------------------------------------
# Функция арифметического кроссовера
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def crossover(parents, crossover_rate):
    childs = []
    parents_size = len(parents)
    for i in range(parents_size):
        parent1_index = random.randint(0, parents_size - 1)
        parent2_index = random.randint(0, parents_size - 1)
        while parent1_index == parent2_index:
            parent2_index = random.randint(0, parents_size - 1)
        if np.random.random() <= crossover_rate:
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
        else:
            childs.append(parents[parent1_index])
            childs.append(parents[parent2_index])
    return childs


# -----------------------------------------------------------
# Функция мутации
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def mutation(children, mutation_rate):
    for i in range(len(children)):
        if np.random.random() <= mutation_rate:
            selected_gen = np.random.randint(0, 3)
            change_value = np.random.randint(0, 1)
            if (change_value == 0):
                children[i, selected_gen] = children[i, selected_gen] - 0.1 * children[i, selected_gen]
            else:
                children[i, selected_gen] = children[i, selected_gen] + 0.1 * children[i, selected_gen]
    return children


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def gen_alg(image, mutation_rate):
    epochs = 60
    populationSize = 50
    k = int(np.round(populationSize * 0.1))
    crossover_rate = 0.8

    global_brightness_value = global_brightness_value_calc(image)

    images_population = generate_population(populationSize)

    for i in range(populationSize):
        images_population[i, 5] = chromosome_improve(images_population[i], image,
                                                     global_brightness_value)

    fitness_values_array = []

    for i in range(epochs):
        images_population[np.isnan(images_population)] = 0
        sorted_population = population_sort(images_population)
        k_best = sorted_population[-k:]

        fitness_values_array.append(k_best[-1, 5])

        selected_parents = np.stack(binary_tournament(sorted_population), axis=0)

        children = np.stack(crossover(selected_parents, crossover_rate), axis=0)

        children = mutation(children, mutation_rate)

        for i in range(len(children)):
            if (children[i, 5] == 0):
                children[i, 5] = chromosome_improve(children[i], image,
                                                    global_brightness_value)

        children[np.isnan(children)] = 0
        sorted_children = population_sort(children)
        best_children = sorted_children[:len(children)-k, :]
        images_population = np.concatenate((k_best, best_children), axis=0)

    final_population = population_sort(images_population)
    best_chromo = final_population[-1]

    # plot_generations_graph(fitness_values_array)

    n = int(best_chromo[4])
    off = n // 2
    new_image_bordered = make_mirror_reflection(image, off)

    return transformaton_calculation(image, new_image_bordered, n, off, global_brightness_value, best_chromo)
