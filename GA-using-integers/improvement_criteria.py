import cv2
import math
import numba
import numpy as np

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
            if (G != 0):
                pix_count = pix_count + 1
            E = E + G
            sobel_img[i, j] = G

    return E, pix_count


# -----------------------------------------------------------
# Фитнес-функция
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def fitness_function(image):
    E, pix_count = sum_intensity(image)
    fitness_value = math.log(math.log(E) * pix_count)
    return fitness_value
