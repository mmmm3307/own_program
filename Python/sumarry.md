## 主要目标
在不降低传统DCT变换去噪算法性能的同时大幅降低其计算量，使其满足在FPGA上实现的要求。
## 工作内容
- 检索传统图像去噪领域论文与仿真代码、对比去噪效果
- 检索有关DCT优化方向论文
- 复现RR-DCT
## 文献复现
《Randomized Redundant DCT Efficent Denoising by Using Random Subsampling of DCT Patches》
1. 主要思想
![](https://raw.githubusercontent.com/mmmm3307/ob_img/main/202407231623119.png)
2. 仿真结果
![](https://raw.githubusercontent.com/mmmm3307/ob_img/main/gray.png)
![](https://raw.githubusercontent.com/mmmm3307/ob_img/main/denoised.png)
3. 现存问题
- 复现代码的运行速度依旧达不到论文中的实验结果
- 去噪图像的四周会出现不规则的小黑边
- 在FPGA上实现的话需要DCT Basis矩阵中的值近似为二进制数，而这样的近似会导致图像经过DCT与IDCT变换后被损坏。
