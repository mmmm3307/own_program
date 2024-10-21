import cv2

# 读取RGB图像
image = cv2.imread('lena.png')

# 将RGB图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示结果
# cv2.imshow('Gray Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存灰度图
cv2.imwrite('gray_lena.jpg', gray_image)
