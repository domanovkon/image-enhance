import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import numba
import math
from numba import njit
import skimage.measure

def histogram_equalization(path):
    img = cv.imread(path, 0)
    cv.imshow("before", img)

    # calculate hist
    hist, bins = np.histogram(img, 256)

    # calculate cdf
    cdf = hist.cumsum()

    # plot hist
    plt.plot(hist, 'r')

    # remap cdf to [0,255]
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8

    # generate img after Histogram Equalization
    img2 = np.zeros((384, 495, 1), dtype=np.uint8)
    img2 = cdf[img]

    hist2, bins2 = np.histogram(img2, 256)
    cdf2 = hist2.cumsum()
    plt.plot(hist2, 'g')
    cv.imshow("after", img2)
    plt.show()
    cv.waitKey(0)
    calculate_metrics(img, img2)


def calculate_metrics(initial, improved):
    E_init, pix_count_init = sum_intensity(initial)
    LQ_init = level_of_adaptation(initial)
    entropy_init = measure_of_entropy(initial)
    print("Исходное изображение")
    print("Уровень адаптации по яркости", LQ_init)
    print("Количество краевых пикселей", pix_count_init)
    print("Суммарная интенсивность краевых писелей", E_init)
    print("Мера энтропии", entropy_init)

    print("---------------")

    E_imp, pix_count_imp = sum_intensity(improved)
    LQ_imp = level_of_adaptation(improved)
    entropy_imp = measure_of_entropy(improved)
    print("Улучшенное изображение")
    print("Уровень адаптации по яркости", LQ_imp)
    print("Количество краевых пикселей", pix_count_imp)
    print("Суммарная интенсивность краевых писелей", E_imp)
    print("Мера энтропии", entropy_imp)

@njit(fastmath=True, cache=True)
def level_of_adaptation(image):
    max_intensity = image.max() / 2
    gl_br_val = np.mean(image)
    LQ = 1 - (math.fabs(gl_br_val - max_intensity) / max_intensity)
    return LQ

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

def measure_of_entropy(image):
    return skimage.measure.shannon_entropy(image)



if __name__ == '__main__':
    # path = "../imgs/11.png"
    # path = "../imgs/camera_man_3.png"
    # path = "../../../../../../Downloads/Telegram Desktop/IMG_20230503_225415_401-05-01-02.jpeg"
    path = "../../../../../../Downloads/Telegram Desktop/IMG_20230510_225637_922-03.jpeg"
    histogram_equalization(path)
