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

    # Изображение с зеркальным отражением границ относительно каждого края
    # Сделано для решения проблемы краевых пикселей
    image_bordered = cv2.copyMakeBorder(src=image, top=off, bottom=off, left=off, right=off,
                                        borderType=cv2.BORDER_REFLECT)

    start_time = time.time()

    gen_alg(image, image_bordered)



    # params = []
    # params.append(5)
    # global_brightness_value = global_brightness_value_calc(image)
    # new_image = transformaton_calculation(image, image_bordered, n, off, global_brightness_value, params)
    # print("new")
    # calculate_fintess(new_image)
    # print("old")
    # calculate_fintess(image)



    end_time = time.time() - start_time
    print("Время выполнения --- %s секунд ---" % end_time)

    # cv2.imshow("Improved image", new_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    load_image('../imgs/camera_man_3.png')
