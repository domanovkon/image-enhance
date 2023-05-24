import cv2
import time

from transformation_kernel import *
from improvement_criteria import *
from genetic_algorithm import *


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Initial image", image)
    cv2.waitKey(0)

    # Размер окрестности
    n = 7
    off = n // 2

    image_bordered = make_mirror_reflection(image, off)

    mutation_rate = 0.07

    start_time = time.time()

    new_image = gen_alg(image, mutation_rate)

    # params = []
    # params.append(7)
    # global_brightness_value = global_brightness_value_calc(image)
    # new_image = transformaton_calculation(image, image_bordered, n, off, global_brightness_value, params)
    # print("new")
    # calculate_fintess(new_image)
    # print("old")
    # calculate_fintess(image)

    end_time = time.time() - start_time
    print("Изображение улучшено")
    print("Время выполнения --- %s секунд ---" % end_time)
    print("---------------")
    cv2.imshow("Improved image", new_image)
    cv2.waitKey(0)

    save_improved_image(new_image, image)

    calculate_metrics(image, new_image)


if __name__ == '__main__':
    path = "../input_images/5.jpeg"
    load_image(path)
