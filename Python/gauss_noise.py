import cv2
import numpy
import math
import numpy.matlib

def Gauss_noise(img, sigma=25):
    noise = numpy.matlib.randn(img.shape) * sigma
    res = img + noise
    return res

if __name__ == '__main__':
    img = cv2.imread('./gray_lena.jpg', cv2.IMREAD_GRAYSCALE)
    noise_img = Gauss_noise(img)
    cv2.imwrite('./noise.jpg', noise_img)