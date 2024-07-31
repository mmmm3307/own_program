import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/ciiv04/program/own_program/Python/test/BM3D/basic/basic2.jpg', cv2.IMREAD_GRAYSCALE)

# 应用拉普拉斯算子
laplacian = cv2.Laplacian(image, cv2.CV_64F)
sharpened_image = cv2.convertScaleAbs(image - laplacian)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
