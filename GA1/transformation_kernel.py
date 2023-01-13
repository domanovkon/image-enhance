import cv2
import math
import numba
import numpy as np
import time
import sys

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
# Функция преобразования отдельного пикселя
# -----------------------------------------------------------
def transformation_calc(pixel_intensity, M, m, sigma):
    a = 1
    b = 0.4
    c = 1
    k = 1
    epsilon = sys.float_info.epsilon
    new_pixel_value = ((k * M) / (sigma + b + epsilon)) * (pixel_intensity - c * m) + (m ** a)
    if new_pixel_value < 0:
        new_pixel_value = 0
    elif new_pixel_value > 255:
        new_pixel_value = 255
    return new_pixel_value



# -----------------------------------------------------------
# Функция преобразования изображения
# -----------------------------------------------------------
# @njit(fastmath=True, cache=True, parallel=True)
def pixel_improvement(image, image_bordered, n, off):
    global_brightness_value = np.mean(image)
    new_image = image.copy()

    initial_image_height = image.shape[0]
    initial_image_width = image.shape[1]

    for i in numba.prange(0, initial_image_height):
        for j in range(0, initial_image_width):
            av_br_value = average_brightness_value_calc(image_bordered, i, j, off)
            st_dev_value = standard_deviation_calc(image_bordered, i, j, off)
            new_pixel_value = transformation_calc(image[i, j], global_brightness_value, av_br_value, st_dev_value)
            new_image[i, j] = int(new_pixel_value)
    cv2.imshow("Improved image", new_image)
    cv2.waitKey(0)
