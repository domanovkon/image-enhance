import cv2
import math
import numba
import numpy as np
import skimage.measure

from skimage.metrics import structural_similarity
from numba import njit


# -----------------------------------------------------------
# Функция для расчета суммарной интенсивности краевых
# пикселей и их количества.
# Для выделения контуров используются маски оператора Собеля.
# -----------------------------------------------------------
@njit(fastmath=True, cache=True, parallel=True)
def sum_intensity(image):
    E = 0
    pix_count = 0
    for i in numba.prange(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            hi = image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1] - image[i - 1, j - 1] - 2 * image[
                i, j - 1] - image[i + 1, j - 1]

            vi = image[i + 1, j + 1] + 2 * image[i + 1, j] + image[i + 1, j - 1] - image[i - 1, j + 1] - 2 * image[
                i - 1, j] - image[i - 1, j - 1]

            G = round(math.sqrt((hi ** 2) + (vi ** 2)))
            if (G > 130):
                pix_count = pix_count + 1
            E = E + G
    return E, pix_count


# -----------------------------------------------------------
# Вычисление меры энтропии изображения
# -----------------------------------------------------------
def measure_of_entropy(image):
    return skimage.measure.shannon_entropy(image)


# -----------------------------------------------------------
# Вычисление уровня адаптации зрительной системы
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def level_of_adaptation(image):
    max_val = image.max()
    if max_val == 0:
        max_val = 1
    middle_of_range = max_val / 2
    gl_br_val = np.mean(image)
    return  1 - (math.fabs(gl_br_val - middle_of_range) / middle_of_range)


# -----------------------------------------------------------
# Получение критериев оценки качества изображения
# -----------------------------------------------------------
def calculate_fintess(image):
    entropy = measure_of_entropy(image)
    E, pix_count = sum_intensity(image)
    LQ = level_of_adaptation(image)
    fitness_value = fitness_function(image, E, pix_count, entropy, LQ)
    # print("LQ", LQ)
    # print("sum int", E)
    # print("pix count" ,pix_count)
    # print("enropy", entropy)
    # print("Fit value", fitness_value)
    return fitness_value


# -----------------------------------------------------------
# Вычисление значения фитнес-функции
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def fitness_function(image, E, pix_count, entropy, LQ):
    return math.log(math.log(E) + math.e) * (pix_count / (image.shape[0] * image.shape[1])) * (math.e ** entropy) * LQ
