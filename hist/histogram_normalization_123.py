import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numba
from numba import njit
import skimage.measure


# histogram normalization
def hist_normalization(img, a=0, b=255):
    # get max and min
    c = img.min()
    d = img.max()

    out = img.copy()

    # normalization
    out = (b - a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)

    return out


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

path = "../imgs/camera_man_3.png"
# path = "../../../../../../Downloads/Telegram Desktop/IMG_20230503_225415_401-05-01-02.jpeg"
path = "../../../../../../Downloads/Telegram Desktop/IMG_20230510_225637_922-03.jpeg"
# path = "../imgs/11.png"
# Read image
img = cv2.imread(path, 0)
# histogram normalization
out = hist_normalization(img)
calculate_metrics(img, out)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.jpg")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.imwrite("out.jpg", out)
cv2.waitKey(0)
