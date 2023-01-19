import cv2
import math
import numba
import numpy as np
import sys

from numba import njit
from improvement_criteria import *

epsilon = sys.float_info.epsilon


# -----------------------------------------------------------
# Расчет сраднего значения яркости изображения
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def global_brightness_value_calc(image):
    return np.mean(image)


# -----------------------------------------------------------
# Расчет сраднего значения яркости в окрестности пикселя
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def average_brightness_value_calc(image, i, j, off):
    i = i + off
    j = j + off
    return np.mean(image[i - off: i + off + 1, j - off: j + off + 1])


# -----------------------------------------------------------
# Расчет среднеквадратического отклонения в окрестности
# с добавлением области.
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def standard_deviation_calc(image, i, j, av_br_val, n, off):
    i = i + off
    j = j + off
    val = 0
    for x in range(i - off, i + off + 1):
        for y in range(j - off, j + off + 1):
            val = val + ((image[x, y] - av_br_val) ** 2)

    black_area = 0
    white_area = 255
    val = val + (n ** 2 * ((black_area - av_br_val) ** 2))
    return math.sqrt(val / (n ** 2) * 2)


# -----------------------------------------------------------
# Функция преобразования отдельного пикселя
#   a ∈ [0 , 1.5]
#   b ∈ [0 , 0.5]
#   c ∈ [0 , 1]
#   k ∈ [0.5 , 1.5]
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def pixel_improvement(pixel_intensity, M, m, sigma, params):
    a = params[0]
    b = params[1]
    c = params[2]
    k = params[3]
    # a = 0.21
    # b = 0.5
    # c = 0.39
    # k = 1.1
    new_pixel_value = int((k * (M / (sigma + b + epsilon))) * (pixel_intensity - c * m) + (m ** a))
    if new_pixel_value < 0:
        new_pixel_value = 0
    elif new_pixel_value > 255:
        new_pixel_value = 255
    return new_pixel_value


# -----------------------------------------------------------
# Функция преобразования изображения
# -----------------------------------------------------------
@njit(fastmath=True, cache=True, parallel=True)
def transformaton_calculation(image, image_bordered, n, off, global_brightness_value, params):
    improved_image = image.copy()

    for i in numba.prange(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            av_br_value = average_brightness_value_calc(image_bordered, i, j, off)
            st_dev_value = standard_deviation_calc(image_bordered, i, j, av_br_value, n, off)
            improved_image[i, j] = pixel_improvement(image[i, j], global_brightness_value, av_br_value, st_dev_value,
                                                     params)
    return improved_image


# -----------------------------------------------------------
# Расчет преобразования и значения фитнес-функции
# -----------------------------------------------------------
def chromosome_improve(params, image, image_bordered, global_brightness_value):
    n = int(params[4])
    off = n // 2

    improved_image = transformaton_calculation(image, image_bordered, n, off, global_brightness_value, params)

    return calculate_fintess(improved_image)
