import random
import math
import cv2
import numpy as np

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

# 生成1920x1080的采样图和采样图LUT
width, height = 1920, 1080
patch_size = 8  # 设置为0以避免复杂性
number_of_lut = 20
d = 30  # 设定mind值
sampling_type = "POISSONDISK"

sampling_maps = generate_sampling_maps(width, height, patch_size, number_of_lut, d, sampling_type)
sampling_map = get_sampling_from_lut(sampling_maps)

sampling_map_np = np.array(sampling_map, dtype=np.uint8)
cv2.imshow("Sampling Map", sampling_map_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
