import cv2
from enhance import *
from genetic_algorithm import *
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('uint8')
    cv2.imshow("Initial image", image)
    cv2.waitKey(0)
    gen_alg(image)


def show_image(image, rows, cols):
    final_image = convert_1D_array_to_image(image, rows, cols)
    cv2.imshow("Final image", final_image)
    cv2.waitKey(0)
    cv2.imwrite("../result/Final_image.jpg", final_image)

    unsharp_image_filter = Image.fromarray(final_image.astype('uint8'))
    new_image = unsharp_image_filter.filter(ImageFilter.UnsharpMask(radius=2, percent=100))
    open_cv_image = np.array(new_image)
    cv2.imshow("Afger unsharp filter",open_cv_image)
    cv2.waitKey(0)
    return 0


if __name__ == '__main__':
    load_image('../imgs/camera_man_3.png')
