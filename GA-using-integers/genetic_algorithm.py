import numba
import numpy as np

from numba import njit
from enhance import *
from improvement_criteria import *
from main import *


# -----------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------
def gen_alg(initial_image):
    # Размер популяции
    population_size = 50

    # Коэффициент кроссовера
    crossover_ratio = 0.2

    # Количество кроссоверов
    crossovers_count = 8

    rows = initial_image.shape[0]
    cols = initial_image.shape[1]

    max_gray_level_value = get_max_gray_level(initial_image)

    all_gray_levels = get_all_gray_levels(initial_image)

    gray_levels_count = len(all_gray_levels)

    initial_population = generate_initial_population(population_size, gray_levels_count, max_gray_level_value)

    image_1D = convert_image_to_1D_array(initial_image, rows, cols)

    fitness_values = []
    enhanced_array = []

    fitness_values, enhanced_array = calculate_fintes_to_initial_population(initial_population, all_gray_levels,
                                                                            image_1D, rows, cols)

    previous_greatest_fitnes_value = 0

    while True:
        new_population, new_fit_vals = crossover(enhanced_array, fitness_values, rows, cols, crossovers_count, population_size,
                                   crossover_ratio)

        fitness_values = new_fit_vals

        # print(len(fitness_values))

        enhanced_array.extend(new_population)

        # Выбираем лучшее значение пригодности
        greatest_fitnes_value = max(fitness_values)
        greatest_fitnes_value_indx = fitness_values.index(greatest_fitnes_value)

        # Уменьшаем размер популяции
        population_size = population_size - len(new_population)

        eps = 0.02 * greatest_fitnes_value

        if greatest_fitnes_value - previous_greatest_fitnes_value < eps or population_size == 1:
            show_image(enhanced_array[greatest_fitnes_value_indx], rows, cols)
            print("Max fitnes value")
            print(greatest_fitnes_value)
            break

        previous_greatest_fitnes_value = greatest_fitnes_value


# -----------------------------------------------------------
# Генерация массивов интенсивностей
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def generate_initial_population(population_size, gray_levels_size, max_gray_level_value):
    population = np.zeros((population_size, gray_levels_size))

    for i in range(0, population_size):
        population[i, :] = np.sort(np.random.randint(0, max_gray_level_value, gray_levels_size))
    return population


# -----------------------------------------------------------
# Расчет преобразования и значения фитнес-функции
# для сгенерированнных массивов интенсивностей
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def calculate_fintes_to_initial_population(population, all_gray_levels, image_1D, rows, cols):
    image_1D_size = rows * cols
    fitness_values = []
    enhanced_array = []
    for value in range(len(population)):
        enhanced = transformation(all_gray_levels, image_1D_size, image_1D, population[value])
        enhanced_image = convert_1D_array_to_image(enhanced, rows, cols)

        enhanced_array.append(enhanced)

        fit_value = fitness_function(enhanced_image)
        fitness_values.append(fit_value)
        # print(value)
        # print(fit_value)
    return fitness_values, enhanced_array


# -----------------------------------------------------------
# Скрещивание
# -----------------------------------------------------------
def crossover(parents, fitness_values, rows, cols, crossovers_count, population_size, crossover_ratio):
    new_population = []
    final_new_population = []
    new_fitness_values = []
    for i in range(crossovers_count):
        firt_parent_index = np.random.randint(0, len(parents))
        second_parent_index = np.random.randint(0, len(parents))

        # Если выбрали одинаковых родителей
        if firt_parent_index == second_parent_index:
            second_parent_index = np.random.randint(0, len(parents))

        first_child, second_child = two_point_cross(parents[firt_parent_index], parents[second_parent_index])
        new_population.append(first_child)
        new_population.append(second_child)

        # Считаем пригодность для потомков
        first_image = convert_1D_array_to_image(first_child, rows, cols)
        first_fit_value = fitness_function(first_image)
        new_fitness_values.append(first_fit_value)

        second_image = convert_1D_array_to_image(second_child, rows, cols)
        second_fit_value = fitness_function(second_image)
        new_fitness_values.append(second_fit_value)

    # Выбираем лучших особей в новой популяции
    for i in range(round(population_size * crossover_ratio)):
        max_fit_value = max(new_fitness_values)
        for count, value in enumerate(new_fitness_values):
            if value == max_fit_value:
                fitness_values.append(value)
                final_new_population.append(new_population[count])
                new_fitness_values.remove(value)

    return final_new_population, fitness_values


# -----------------------------------------------------------
# Двухточенчный кроссинговер. На вход подаем 2-х родителей.
# На выходе получаем 2-х потомков.
# -----------------------------------------------------------
def two_point_cross(first_parent, second_parent):
    size = min(len(first_parent), len(second_parent))

    cross_point1 = np.random.randint(0, size)
    cross_point2 = np.random.randint(0, size - 1)

    if cross_point1 > cross_point2:
        cross_point1, cross_point2 = cross_point2, cross_point1
    else:
        cross_point2 += 1

    # Происходит обмен интенсивностями пикселей между точками
    temp = first_parent.copy()

    first_parent[cross_point1:cross_point2] = second_parent[cross_point1:cross_point2]
    second_parent[cross_point1:cross_point2] = temp[cross_point1:cross_point2]

    return first_parent, second_parent
