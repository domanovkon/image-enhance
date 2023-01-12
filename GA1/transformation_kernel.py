import cv2
import math
import numba
import numpy as np
import time

from numba import njit


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
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def standard_deviation_calc(image, i, j, off):
    i = i + off
    j = j + off
    return np.std(image[i - off: i + off + 1, j - off: j + off + 1])


# -----------------------------------------------------------
# Функция преобразования изображения
# -----------------------------------------------------------
# @njit(fastmath=True, cache=True)
def pixel_improvement(image, image_bordered, n, off):
    global_brightness_value = np.mean(image)
    new_image = image.copy()

    initial_image_height = image.shape[0]
    initial_image_width = image.shape[1]

    for i in numba.prange(0, initial_image_height):
        for j in range(0, initial_image_width):
            av_br_val = average_brightness_value_calc(image_bordered, i, j, off)
            st_br_val = standard_deviation_calc(image_bordered, i, j, off)
