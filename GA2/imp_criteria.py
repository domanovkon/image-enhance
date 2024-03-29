import numpy as np
import math
import skimage.measure

from ench import *


# -----------------------------------------------------------
# Функция для расчета суммарной интенсивности краевых
# пикселей и их количества.
# Для выделения контуров используются маски оператора Собеля.
# -----------------------------------------------------------
@njit(fastmath=True, cache=True, parallel=True)
def sum_intensity(image):
    E = 0
    pix_count = 0
    sobel_img = image.copy()
    for i in numba.prange(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if i == 0 or j == 0 or i == image.shape[0] - 1 or j == image.shape[1] - 1:
                sobel_img[i, j] = 0
                continue
            hi = image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1] - image[i - 1, j - 1] - 2 * image[
                i, j - 1] - image[i + 1, j - 1]

            vi = image[i + 1, j + 1] + 2 * image[i + 1, j] + image[i + 1, j - 1] - image[i - 1, j + 1] - 2 * image[
                i - 1, j] - image[i - 1, j - 1]

            G = round(math.sqrt((hi ** 2) + (vi ** 2)))
            if (G > 130):
                pix_count = pix_count + 1
            E = E + G
            sobel_img[i, j] = G

    return E, pix_count


# -----------------------------------------------------------
# Вычисление количества краевых пикселей изображения
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def calculate_edge_count(image):
    E, pix_count = sum_intensity(image)
    return pix_count


# -----------------------------------------------------------
# Вычисление уровня адаптации зрительной системы
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def level_of_adaptation(image):
    max_intensity = image.max() / 2
    gl_br_val = np.mean(image)
    LQ = 1 - (math.fabs(gl_br_val - max_intensity) / max_intensity)
    return LQ


# -----------------------------------------------------------
# Вычисление значения фитнес-функции для каждой хромосомы
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def calculate_fintess_value(gray_levels, chromosome, image_array):
    image = create_enhanced_image(gray_levels, chromosome, image_array)
    E, pix_count = sum_intensity(image)
    LQ = level_of_adaptation(image)
    fitness_value = math.log(math.log(E) * pix_count) * LQ
    return fitness_value


# -----------------------------------------------------------
# Вычисляет массив значений фитнес-функции для
# всей популяции
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def fitness_for_all_population(population, gray_levels, img_array):
    fitness = []
    for chromosome in population:
        fitness.append(calculate_fintess_value(gray_levels, chromosome, img_array))
    return np.array(fitness)


def image_comparison(image, hist_equalized_img, improved_image):
    initial_sum_intensity, initial_edge_count = sum_intensity(image)
    hist_sum_intensity, hist_edge_count = sum_intensity(hist_equalized_img)
    improved_sum_intensity, improved_edge_count = sum_intensity(improved_image)
    print("Количество ребер исходного изображения: ", initial_edge_count)
    print("Количество ребер после эквализации гистограммы: ", hist_edge_count)
    print("Количество ребер улучшенного изображения: ", improved_edge_count)
    return initial_edge_count, hist_edge_count, improved_edge_count


def measure_of_entropy(image):
    return skimage.measure.shannon_entropy(image)


def calculate_metrics(initial, improved):
    E_init, pix_count_init = sum_intensity(initial)
    LQ_init = level_of_adaptation(initial)
    ent_init = measure_of_entropy(initial)
    print("Исходное изображение")
    print("Уровень адаптации по яркости", LQ_init)
    print("Количество краевых пикселей", pix_count_init)
    print("Суммарная интенсивность краевых писелей", E_init)
    print("Энтропия", ent_init)

    print("---------------")

    E_imp, pix_count_imp = sum_intensity(improved)
    LQ_imp = level_of_adaptation(improved)
    ent_imp = measure_of_entropy(improved)
    print("Улучшенное изображение")
    print("Уровень адаптации по яркости", LQ_imp)
    print("Количество краевых пикселей", pix_count_imp)
    print("Суммарная интенсивность краевых писелей", E_imp)
    print("Энтропия", ent_imp)
