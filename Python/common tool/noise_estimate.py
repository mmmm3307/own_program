import cv2
import numpy as np

def estimate_noise_std(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的中值
    median = cv2.medianBlur(gray, 3)

    # 计算差值图像
    diff = gray - median

    # 计算绝对偏差的中值
    mad = np.median(np.abs(diff))

    # 将MAD转换为标准差的估计值
    noise_std = 1.4826 * mad

    return noise_std

# 读取图像
image = cv2.imread('/home/ciiv04/program/own_program/Python/test/gray/gray3.jpg')

# 估计噪声标准差
noise_std = estimate_noise_std(image)

print(f"Estimated noise standard deviation:1 {noise_std}")
