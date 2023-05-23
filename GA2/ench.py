import numpy as np
import matplotlib.pyplot as plt
import cv2
import numba

from numba import njit


# -----------------------------------------------------------
# Получение общего количества уровней серого в изображении
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def get_all_gray_levels_len(image):
    return len(np.unique(np.asarray(image)))


# -----------------------------------------------------------
# Получение массива интенсивностей и подсчет количества
# уровней серого
# -----------------------------------------------------------
@njit(fastmath=True, cache=True)
def preparation_ga_parameters(image_array):
    gray_levels = np.unique(image_array)
    gray_levels.sort()
    gray_levels_len = len(gray_levels)
    return gray_levels, gray_levels_len


# -----------------------------------------------------------
# Перебираем каждый пиксель и меняем его на улучшенный
# -----------------------------------------------------------
@njit(fastmath=True, cache=True, parallel=True)
def create_enhanced_image(original_gray_levels, enhanced_gray_levels, image_array):
    pixel_map = {}

    # Сохраняем значения интенсивности пикселей в мапу
    for i in range(len(enhanced_gray_levels)):
        pixel_map[original_gray_levels[i]] = enhanced_gray_levels[i]

    enhanced_image = np.empty(image_array.shape)

    for i in numba.prange(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            enhanced_image[i][j] = pixel_map[image_array[i][j]]
    return enhanced_image


# -----------------------------------------------------------
# Эквализация гистограммы. Необходимо для исследования
# работы разработанного метода
# -----------------------------------------------------------
def equalize_histogram(img_array):
    gray_levels, freq = np.unique(img_array, return_counts=True)
    N = len(gray_levels)
    histogram = enumerate(freq)
    nk = []
    pdf = []
    cdf = []
    final_grey_levels = []
    tp_cdf = 0
    total = sum(freq)
    for i in range(N):
        nk.append(freq[i])
        pdf.append(freq[i] / total)
        cdf.append(tp_cdf + pdf[i])
        tp_cdf = cdf[i]
        final_grey_levels.append(round((N - 1) * cdf[i]))
    return create_enhanced_image(gray_levels, final_grey_levels, img_array)


# -----------------------------------------------------------
# Выводим изображения после преобразований
# -----------------------------------------------------------
def show_final_images(image, final_image, random_image, equalized_image, file_name):
    f, axarr = plt.subplots(2, 2, figsize=(10, 10))

    axarr[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axarr[0, 0].set_title("Исходое изображение")

    axarr[0, 1].imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].set_title("Эквализация гистограммы")

    axarr[1, 0].imshow(random_image, cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].set_title("Случайно выбрано из популяции")

    axarr[1, 1].imshow(final_image, cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].set_title("Генетический алгоритм")

    plt.savefig(file_name)


# -----------------------------------------------------------
# Расчет зависимости коэффициента от фитнес-функции
# -----------------------------------------------------------
def counting_statistics(rate_name, solutions):
    rate = []
    fitness = []
    edges_count = []
    for solution in solutions:
        rate.append(solution[rate_name])
        fitness.append(solution['fitness'])
        edges_count.append(solution['edges_count'])
    return rate, fitness


# -----------------------------------------------------------
# Строит график зависимости кроссовера/мутации от
# максимального значения фитнес-функции
# -----------------------------------------------------------
def show_graph(rate, fitness, label):
    name = ""
    num = 0
    if label == 'Коэффициент кроссовера':
        name = 'result/fitnes-crossover.png'
    else:
        num = 1
        name = 'result/fitnes-mutation.png'
    plt.figure(figsize=(10, 10))
    plt.title("Зависимость")
    plt.plot(rate, fitness)
    plt.xlabel(label)
    plt.ylabel('Фитнес-функция')
    plt.savefig(name)


# -----------------------------------------------------------
# Строит график зависимости поколения от значения
# фитнес-функции
# -----------------------------------------------------------
def plot_generations_graph(fitness_values_array):
    plt.figure(figsize=(10, 10))
    plt.plot([i for i in range(len(fitness_values_array))], fitness_values_array)
    plt.rc('font', size=30)
    plt.tick_params(labelsize=18)
    plt.xlabel("Поколение", fontsize=22)
    plt.ylabel("Фитнес-функция", fontsize=22)
    plt.savefig('result/generations-fitness.png')


# -----------------------------------------------------------
# Сохраняет итоговое изображение
# -----------------------------------------------------------
def save_improved_image(improved_image, initial_image):
    plt.rc('font', size=12)
    plt.tick_params(labelsize=18)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle('Метод подбора интенсивностей (Глобальный подход)')

    ax1.imshow(initial_image, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Исходое изображение")

    ax2.imshow(improved_image, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Улучшенное изображение")

    plt.savefig('result/improved.png')
