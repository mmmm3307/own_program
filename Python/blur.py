import cv2
import numpy as np

img = cv2.imread('./img/denoising1.bmp', cv2.IMREAD_GRAYSCALE)
blur_img = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imwrite('./blurred.png',blur_img)

