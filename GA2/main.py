import numpy as np
import cv2
import copy
import numba
import time

from numba import njit

from gen_alg import *
from ench import *
from imp_criteria import *

population_size = 100

crossover_rate = 0.3

mutation_rate = 0.1

generations_count = 80

research_mode = True


# -----------------------------------------------------------
# Загрузка исходного изображения
# -----------------------------------------------------------
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('uint8')
    cv2.imshow("Initial image", image)
    cv2.waitKey(0)
    return image


def research_enhancement(path):
    image = load_image(path)

    all_gray_levels_len = get_all_gray_levels_len(image)

    start_time = time.time()

    population = generate_initial_population(all_gray_levels_len, population_size)

    cross_solutions = []
    print("Расчет лучшего коэффициента кроссовера")
    for c_rate in np.arange(0.1, 1, 0.1):
        print("Коэффициент %.1f%%" % (c_rate * 100))
        improved_image, max_fitness = genetic_algorithm(image, population_size, c_rate, mutation_rate,
                                                        generations_count, copy.deepcopy(population), False)

        calculated_parameters = {"image": improved_image, "crossover_rate": c_rate, "fitness": max_fitness,
                                 "edges_count": calculate_edge_count(improved_image)}

        cross_solutions.append(calculated_parameters)

    best_fitness_cross_solution = max(cross_solutions, key=lambda x: x['fitness'])

    image_array = np.asarray(image)

    # Получаем массив интенсивностей пикселей
    gray_levels, gray_levels_len = preparation_ga_parameters(image_array)

    example_image_from_initial_population = create_enhanced_image(gray_levels, numba.typed.List(population[0]),
                                                                  image_array)

    hist_equalized_img = equalize_histogram(image_array)

    improved_image = best_fitness_cross_solution['image']

    show_final_images(image, improved_image, example_image_from_initial_population, hist_equalized_img,
                      'result/crossover-research.png')

    print("Лучший коэффициент кроссовера: %.1f" % best_fitness_cross_solution['crossover_rate'])
    image_comparison(image, hist_equalized_img, improved_image)

    mutation_solutions = []
    crossover_rate = 0.3
    print("Расчет лучшей вероятности мутации")
    for m_rate in np.arange(0.1, 1, 0.1):
        print("Вероятность %.1f%%" % (m_rate * 100))
        improved_image, max_fitness = genetic_algorithm(image, population_size, crossover_rate, m_rate,
                                                        generations_count, copy.deepcopy(population), False)

        calculated_parameters = {"image": improved_image, "mutation_rate": m_rate, "fitness": max_fitness,
                                 "edges_count": calculate_edge_count(improved_image)}

        mutation_solutions.append(calculated_parameters)

    best_fitness_mutation_solution = max(mutation_solutions, key=lambda x: x['fitness'])

    improved_image = best_fitness_mutation_solution['image']

    show_final_images(image, improved_image, example_image_from_initial_population, hist_equalized_img,
                      'result/mutation-research.png')

    print("Лучшая вероятность мутации: %.1f" % best_fitness_mutation_solution['mutation_rate'])

    image_comparison(image, hist_equalized_img, improved_image)

    c_rate = []
    fitness = []

    c_rate, fitness = counting_statistics('crossover_rate', cross_solutions)
    show_graph(c_rate, fitness, 'Коэффициент кроссовера')

    m_rate = []
    fitness = []

    m_rate, fitness = counting_statistics('mutation_rate', mutation_solutions)
    show_graph(m_rate, fitness, 'Вероятность мутации')

    # Получение лучшего коэффициента кроссовера и вероятности мутации
    best_crossover_rate = max(cross_solutions, key=lambda x: x['fitness'])['crossover_rate']
    best_mutation_rate = max(mutation_solutions, key=lambda x: x['fitness'])['mutation_rate']
    final_generations_count = 100

    improved_image, max_fitness = genetic_algorithm(image, population_size, best_crossover_rate, best_mutation_rate,
                                                    final_generations_count, copy.deepcopy(population), True)

    end_time = time.time() - start_time
    print("Время выполнения --- %s секунд ---" % end_time)

    save_improved_image(improved_image, image)


def enhancement(path):
    image = load_image(path)

    all_gray_levels_len = get_all_gray_levels_len(image)

    start_time = time.time()

    population = generate_initial_population(all_gray_levels_len, population_size)

    final_generations_count = 100

    improved_image, max_fitness = genetic_algorithm(image, population_size, crossover_rate, mutation_rate,
                                                    final_generations_count, copy.deepcopy(population), True)

    end_time = time.time() - start_time
    print("Время выполнения --- %s секунд ---" % end_time)

    save_improved_image(improved_image, image)


if __name__ == '__main__':
    file_name = '../imgs/Albert-Einstein.jpg'
    if research_mode == True:
        research_enhancement(file_name)
    else:
        enhancement(file_name)
    print("Изображение улучшено")
