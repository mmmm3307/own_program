import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def gelu(x):
    """GELU的精确公式"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_approx(x):
    """GELU的近似公式"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 生成输入数据
x = np.linspace(-5, 5, 1000)
y_exact = gelu(x)
y_approx = gelu_approx(x)

# 绘制GELU函数
plt.figure(figsize=(8, 6))
plt.plot(x, y_exact, label='GELU 精确公式', color='blue')
plt.plot(x, y_approx, label='GELU 近似公式', color='red', linestyle='--')
plt.title('GELU 激活函数')
plt.xlabel('输入值 x')
plt.ylabel('GELU(x)')
plt.legend()
plt.grid(True)
plt.show()
