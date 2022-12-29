import cv2
import numpy as np
import numba

from numba import njit


# -----------------------------------------------------------
# Считает максимальный уровень серого в изображении
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def get_max_gray_level(image):
    return max(np.unique(image))


# -----------------------------------------------------------
# Возвращает массив всех уровней серого в изображении
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def get_all_gray_levels(image):
    return np.unique(image)


# -----------------------------------------------------------
# Преобразование изображения в одномерный
# массив интенсивности пикселей
# -----------------------------------------------------------
@njit(fastmath=True, cache=False)
def convert_image_to_1D_array(image, rows, cols):
    return np.reshape(image, rows * cols)


# -----------------------------------------------------------
# Преобразование изображения в одномерный
# массив интенсивности пикселей
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def convert_1D_array_to_image(image, rows, cols):
    return np.reshape(image, (rows, cols))


# -----------------------------------------------------------
# Функция расчета изображения для каждой хромосомы
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def transformation(all_gray_levels, image_1D_size, image_1D, value):
    image_1D_copy = image_1D.copy()

    # Делаем подстановку интесивности пикселей из сгенерированной хромосомы в исходное изображение
    # Первый уровень серого заменяется значением первого уровня серого хромосомы и т.д.
    for count, color in enumerate(all_gray_levels):
        for j in range(image_1D_size):
            if image_1D_copy[j] == color:
                image_1D_copy[j] = value[count]

    return image_1D_copy
