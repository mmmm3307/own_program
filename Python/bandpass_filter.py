import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# 读取图像并转换为灰度图像
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 获取图像尺寸
rows, cols = img.shape

# 对图像进行2D FFT变换
f = fft2(img)
f_shifted = fftshift(f)  # 将频谱的零频点移动到中心

# 创建带通滤波器
def bandpass_filter(shape, d0, d1):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # 中心位置
    mask = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            # 计算每个点到中心的距离
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if d0 <= distance <= d1:
                mask[i, j] = 1  # 距离在范围内的部分保留

    return mask

# 设置带通滤波器的频率范围
d0 = 30  # 最小距离
d1 = 80  # 最大距离

# 生成带通滤波器
bandpass = bandpass_filter((rows, cols), d0, d1)

# 应用带通滤波器
filtered_shifted = f_shifted * bandpass

# 将频谱移回并进行逆FFT变换
filtered = ifftshift(filtered_shifted)
img_back = np.abs(ifft2(filtered))

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Filtered Image')
plt.show()
