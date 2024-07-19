import math
import random
import cv2
import numpy as np


'''
dct basis matrix
'''
DCTbasis=[[0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738],
[0.4903926402016152, 0.4157348061512726, 0.27778511650980114, 0.09754516100806417, -0.0975451610080641, -0.277785116509801, -0.4157348061512727, -0.4903926402016152],
[0.46193976625564337, 0.19134171618254492, -0.19134171618254486, -0.46193976625564337, -0.4619397662556434, -0.19134171618254517, 0.191341716182545, 0.46193976625564326],        
[0.4157348061512726, -0.0975451610080641, -0.4903926402016152, -0.2777851165098011, 0.2777851165098009, 0.4903926402016152, 0.09754516100806439, -0.41573480615127256],
[0.3535533905932738, -0.35355339059327373, -0.35355339059327384, 0.3535533905932737, 0.35355339059327384, -0.35355339059327334, -0.35355339059327356, 0.3535533905932733],       
[0.27778511650980114, -0.4903926402016152, 0.09754516100806415, 0.4157348061512728, -0.41573480615127256, -0.09754516100806401, 0.4903926402016153, -0.27778511650980076],       
[0.19134171618254492, -0.4619397662556434, 0.46193976625564326, -0.19134171618254495, -0.19134171618254528, 0.46193976625564337, -0.4619397662556432, 0.19134171618254478],       
[0.09754516100806417, -0.2777851165098011, 0.4157348061512728, -0.4903926402016153, 0.49039264020161527, -0.4157348061512725, 0.27778511650980076, -0.09754516100806429]]

def dct_1d(data, basis):
    """ 8x1 1D DCT """
    result = [0.0] * 8
    for u in range(8):
        sum_val = 0.0
        for x in range(8):
            sum_val += data[x] * basis[u][x]
        result[u] = sum_val
    return result

def idct_1d(data, basis):
    """ 8x1 1d IDCT"""
    result = [0.0] * 8
    for x in range(8):
        sum_val = 0.0
        for u in range(8):
            sum_val += data[u] * basis[u][x]
        result[x] = sum_val
    return result

def dct_2d(patch, basis):
    """ 8x8 2D DCT """
    # 对每一行进行8x1 1维 DCT
    temp = [dct_1d(row, basis) for row in patch] 
    # 对结果进行转置
    transposed = [[temp[j][i] for j in range(8)] for i in range(8)] 
    # 对转置后的结果的每一行进行8x1 1维 DCT
    temp2 = [dct_1d(row, basis) for row in transposed]   
    # 对结果进行转置
    result = [[temp2[j][i] for j in range(8)] for i in range(8)]  

    return result

def idct_2d(patch, basis):
    """ 8x8 2D IDCT """
    # 与DCT一样
    temp = [idct_1d(row, basis) for row in patch]   
    transposed = [[temp[j][i] for j in range(8)] for i in range(8)]    
    temp2 = [idct_1d(row, basis) for row in transposed]    
    result = [[temp2[j][i] for j in range(8)] for i in range(8)]
    
    return result

'''
def dct_2d(patch, basis):
    """8x8 2D DCT """
    result = [[0.0 for _ in range(8)] for _ in range(8)]
    for u in range(8):
        for v in range(8):
            sum_val = 0.0
            for x in range(8):
                for y in range(8):
                    sum_val += patch[x][y] * basis[u][x] * basis[v][y]
            result[u][v] = sum_val
    return result

def idct_2d(patch, basis):
    """8x8 2D IDCT"""
    result = [[0.0 for _ in range(8)] for _ in range(8)]
    for x in range(8):
        for y in range(8):
            sum_val = 0.0
            for u in range(8):
                for v in range(8):
                    sum_val += patch[u][v] * basis[u][x] * basis[v][y]
            result[x][y] = sum_val
    return result
'''

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Grid:
    def __init__(self, cellsize, width, height):
        self.cellsize = cellsize
        self.width = int(width / cellsize) + 1
        self.height = int(height / cellsize) + 1
        self.bpt = [[] for _ in range(self.width * self.height)]

    def image_to_grid(self, pt):
        gx = int(pt.x / self.cellsize)
        gy = int(pt.y / self.cellsize)
        return Point(gx, gy)

    def set(self, pt):
        gpt = self.image_to_grid(pt)
        self.bpt[gpt.x + self.width * gpt.y].append(pt)

def generate_random_point_around(pt, mind):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    radius = mind * (1 + r1)
    angle = 2 * math.pi * r2
    x = int(pt.x + radius * math.cos(angle))
    y = int(pt.y + radius * math.sin(angle))
    return Point(x, y)

def in_kernel(pt, width, height):
    return 0 <= pt.x < width and 0 <= pt.y < height

def distance(pt1, pt2):
    return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

def in_neighbourhood(grid, pt, mind):
    gpt = grid.image_to_grid(pt)
    for y in range(-2, 3):
        if 0 <= y + gpt.y < grid.height:
            for x in range(-2, 3):
                if 0 <= x + gpt.x < grid.width:
                    ppt = Point(x + gpt.x, y + gpt.y)
                    cell = grid.bpt[ppt.x + grid.width * ppt.y]
                    for bpt_ in cell:
                        if distance(bpt_, pt) < mind:
                            return False
    return True

def set_poisson_disk(kernel, mind):
    '''possion sample'''
    height, width = len(kernel), len(kernel[0])
    cellsize = mind / math.sqrt(2)
    grid = Grid(cellsize, width, height)
    proc = []

    first = Point(random.randint(0, width - 1), random.randint(0, height - 1))
    proc.append(first)
    kernel[first.y][first.x] = 255
    grid.set(first)

    while proc:
        pt = proc.pop(0)
        for _ in range(30):
            newpt = generate_random_point_around(pt, mind)
            if in_kernel(newpt, width, height) and in_neighbourhood(grid, newpt, mind):
                proc.append(newpt)
                kernel[newpt.y][newpt.x] = 255
                grid.set(newpt)

def set_sampling_map(width, height, sampling_type, d):
    height = height - 8
    width = width - 8
    kernel = [[0 for _ in range(width)] for _ in range(height)]
    if sampling_type == "FULL":
        for y in range(height):
            for x in range(width):
                kernel[y][x] = 255
    elif sampling_type == "POISSONDISK":
        set_poisson_disk(kernel, d)
    return kernel

def generate_sampling_maps(width, height, patch_size, number_of_lut, d, sampling_type):
    sampling_maps = []
    for _ in range(number_of_lut):
        sampling_map = set_sampling_map(width + patch_size, height + patch_size, sampling_type, d)
        sampling_maps.append(sampling_map)
    return sampling_maps

def get_sampling_from_lut(sampling_maps):
    return random.choice(sampling_maps)

def extract_patches(image, sampling_map, patch_size):
    patches = []
    coords = []
    height, width = image.shape[:2]
    for y in range(height - patch_size + 1):
        for x in range(width - patch_size + 1):
            if sampling_map[y][x] == 255:
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coords.append((y, x))
    return patches, coords


def hard_threshold(matrix, threshold):
    N = len(matrix)
    M = len(matrix[0])
    thresholded_matrix = [[0 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            # print(f"Processing element [{i}][{j}]: {matrix[i][j]}")  
            if isinstance(matrix[i][j], (int, float)):
                if abs(matrix[i][j]) < threshold:
                    thresholded_matrix[i][j] = 0
                else:
                    thresholded_matrix[i][j] = matrix[i][j]
            else:
                raise ValueError(f"Element [{i}][{j}] is not a number: {matrix[i][j]}")
    return thresholded_matrix

def patches_to_image(patches, coords, width, height, patch_size):
    size1 = width * height

    # 初始化图像和权重数组
    im = [0.0] * size1
    im_weight = [0.0] * size1

    # 遍历所有的patch及其对应的坐标
    for patch, (y, x) in zip(patches, coords):
            for jp in range(patch_size):
                for ip in range(patch_size):
                    im_idx = (y + jp) * width + (x + ip)
                    im[im_idx] += patch[jp][ip]
                    im_weight[im_idx] += 1

    # 通过权重归一化图像
    for i in range(size1):
        if im_weight[i] != 0:
            im[i] /= im_weight[i]

    # 将图像重塑为原始维度
    result_image = [[0.0 for _ in range(width)] for _ in range(height)]
    for j in range(height):
        for i in range(width):
            im_idx =  j * width + i
            result_image[j][i] = im[im_idx]

    return result_image

# def hard_threshold(matrix, threshold):
#     thresholded_patch = matrix.copy()  # 复制原始矩阵以避免修改原始数据
#     thresholded_patch[abs(matrix) < threshold] = 0
#     return thresholded_patch

# 生成1920x1080的采样图和采样图LUT
width, height  = 1920, 1080
patch_size = 8
number_of_lut = 1   #生成查找表个数
d = 3  # 设定mind值
sampling_type = "POISSONDISK"

sampling_maps = generate_sampling_maps(width, height, patch_size, number_of_lut, d, sampling_type)
sampling_map = get_sampling_from_lut(sampling_maps) 

# input image 
image = cv2.imread('./denoising1.bmp')
# print(f'imagebeforegray shape:{image.shape},type:{type(image)}')
image = 0.299*image[:, :,0] + 0.587*image[:, :,1] + 0.114*image[:, :,2]
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("./gray.png",image)
# print(f'imageaftergray shape:{image.shape},type:{type(image)}')

# 提取8x8的patches
patches, coords = extract_patches(image, sampling_map, patch_size)

# DCT变换，硬阈值处理，IDCT变换
threshold = 50  # 设定硬阈值
denoised_patches = []
for patch in patches:
    dct_patch = dct_2d(patch,DCTbasis)
    thresholded_patch = hard_threshold(dct_patch, threshold)
    idct_patch = idct_2d(thresholded_patch,DCTbasis)
    denoised_patches.append(idct_patch)

denoised_image = patches_to_image(denoised_patches,coords,width, height, patch_size)
denoised_image = np.array(denoised_image)

# print(type(denoised_image))
# print(denoised_image.shape)

cv2.imwrite("./denoised.png",denoised_image)




  
  
  
  
  
  