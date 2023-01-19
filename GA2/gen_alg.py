import numpy as np
import sys
import random

from random import randint
from numpy.random import default_rng

from ench import *
from imp_criteria import *

rng = default_rng()
np.random.seed(0)


# -----------------------------------------------------------
# Генерация начальной популяции
# -----------------------------------------------------------
def generate_initial_population(all_gray_levels_len, population_size):
    population = []
    for i in range(population_size):
        chromosome = [randint(1, 244) for _ in range(all_gray_levels_len - 2)]
        chromosome.extend([0, 255])
        chromosome.sort()
        population.append(chromosome)
    return np.array(population)


# -----------------------------------------------------------
# Вычисляем общую пригодность популяции и каждой отдельной
# хромосомы
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def probability_calculation(population, fitness):
    population_fitness = sum(fitness)
    chromosome_probabilities = [fitness[i] / population_fitness for i, chromosome in enumerate(population)]
    return population_fitness, chromosome_probabilities


# -----------------------------------------------------------
# Формируем выборку родителей случайным образом
# -----------------------------------------------------------
def selection(population, child_count):
    parents = rng.choice(population, p=None, size=child_count, replace=False)
    return parents


# -----------------------------------------------------------
# Выбираем случайным образом две точки
# [от 0 до количества уровней серого в хромосоме]
# -----------------------------------------------------------
def generate_cross_points(begin, end, count):
    return np.sort(rng.choice(np.arange(begin, end + 1), size=count, replace=False))


# -----------------------------------------------------------
# Считаем индексы родителей для скрещивания
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def get_parent_indices(k, parents_count):
    first_parent_index = (2 * k) % parents_count
    second_parent_index = (2 * k + 1) % parents_count
    return first_parent_index, second_parent_index


# -----------------------------------------------------------
# Двухточечный кроссовер
# -----------------------------------------------------------
def crossover(parents, gray_levels_len, child_count, population):
    for k in range(child_count):
        cross_points = generate_cross_points(0, gray_levels_len, 2)

        first_parent_index, second_parent_index = get_parent_indices(k, parents.shape[0])

        first_child = np.concatenate((parents[first_parent_index][:cross_points[0]],
                                      parents[second_parent_index][cross_points[0]:cross_points[1]],
                                      parents[first_parent_index][cross_points[1]:]))

        second_child = np.concatenate((parents[second_parent_index][:cross_points[0]],
                                       parents[first_parent_index][cross_points[0]:cross_points[1]],
                                       parents[second_parent_index][cross_points[1]:]))
        first_child.sort()
        second_child.sort()
        population = np.vstack([population, first_child, second_child])
    return population


# -----------------------------------------------------------
# Случайным образом выбираем и вставляем значение
# между предыдущим и следующим геном
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def chromosome_replacement(population, chromosome, chromosome_index, mutated_genes_indices):
    for index in mutated_genes_indices:
        chromosome[index] = np.int64(random.randint(chromosome[index - 1].item(), chromosome[index + 1].item()))

    population[chromosome_index] = chromosome
    return population


# -----------------------------------------------------------
# Мутация
# 5 % генов хромосомы подвергаются мутации
# -----------------------------------------------------------
def mutation(mutation_rate, population):
    chromosome_len = len(population[0])

    for chromosome_index, chromosome in enumerate(population):
        if np.random.random() < mutation_rate:
            mutated_genes_count = int(0.05 * chromosome_len)
            # Случайным образом выбираем гены в хромосоме
            # Рассматриваем диапазон [1; n-1]
            mutated_genes_indices = np.sort(
                rng.choice(np.arange(1, chromosome_len - 1), size=mutated_genes_count, replace=False))

            population = chromosome_replacement(population, chromosome, chromosome_index, mutated_genes_indices)

    return population


# -----------------------------------------------------------
# Получение изображения из массива интенсивностей с
# наибольшим значением фитнес-функции
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def get_best_image(new_population, fitness, gray_levels, image_array):
    max_fitnes_value_index = np.where(fitness == max(fitness))

    improved_image_array = new_population[max_fitnes_value_index]
    improved_image_array = np.reshape(improved_image_array, gray_levels.shape)
    improved_image = create_enhanced_image(gray_levels, improved_image_array, image_array)
    return improved_image, max(fitness)


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def genetic_algorithm(image, population_size, crossover_rate, mutation_rate,
                      generations_count, population, is_graph_plot):
    num_individuals = population_size

    # Подсчитываем количество новых особей в каждой популяции
    child_count = int(crossover_rate * num_individuals)

    image_array = np.asarray(image)

    # Получаем массив интенсивностей пикселей
    gray_levels, gray_levels_len = preparation_ga_parameters(image_array)

    new_population = population

    fitness_values_array = []

    for generation in (range(generations_count)):
        fitness = fitness_for_all_population(new_population, gray_levels, image_array)

        fitness_values_array.append(max(fitness))

        parents = selection(new_population, child_count)

        new_population = crossover(parents, gray_levels_len, child_count, new_population)

        new_population = mutation(mutation_rate, new_population)

    if is_graph_plot:
        plot_generations_graph(generations_count, fitness_values_array)

    print("---------------")

    fitness = fitness_for_all_population(new_population, gray_levels, image_array)

    return get_best_image(new_population, fitness, gray_levels, image_array)
