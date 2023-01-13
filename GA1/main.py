import cv2
import time

from transformation_kernel import pixel_improvement


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Initial image", image)
    cv2.waitKey(0)

    # Размер окрестности
    n = 10
    off = n // 2

    # Изображение с зеркальным отражением границ относительно каждого края
    # Сделано для решения проблемы краевых пикселей
    image_bordered = cv2.copyMakeBorder(src=image, top=off, bottom=off, left=off, right=off,
                                        borderType=cv2.BORDER_REFLECT)

    start_time = time.time()

    pixel_improvement(image, image_bordered, n, off)

    end_time = time.time() - start_time
    print("Время выполнения --- %s секунд ---" % end_time)


if __name__ == '__main__':
    load_image('../imgs/camera_man_2.jpeg')
