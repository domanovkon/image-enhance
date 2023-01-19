import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import GA1
from GA1 import improvement_criteria

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
    improvement_criteria.calculate_fintess(img2)
    cv.imshow("after", img2)
    plt.show()
    cv.waitKey(0)


if __name__ == '__main__':
    histogram_equalization("../imgs/low_contrast.jfif")
