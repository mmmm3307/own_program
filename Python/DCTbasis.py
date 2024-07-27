import math
import numpy as np

def generate_dct_basis(n):
    """Generate an n x n DCT basis matrix."""
    basis = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0:
                basis[i][j] = math.sqrt(1 / n)
            else:
                basis[i][j] = math.sqrt(2 / n) * math.cos(math.pi * i * (2 * j + 1) / (2 * n))
    return basis
'''
# Generate an 8x8 DCT basis matrix
dct_basis_8x8 = generate_dct_basis(8)

transpose_dct_basis_8x8 = np.transpose(dct_basis_8x8)

result = np.dot(transpose_dct_basis_8x8,dct_basis_8x8)

print(result)
'''

DCTbasis = [[0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706]]

DCTbasis_bin = [[0.5000, 0.5000, 0.5000, 0.5000],
                [0.625, 0.28125, -0.28125, -0.6533],
                [0.5000, -0.5000, -0.5000, 0.5000],
                [0.28125, -0.625, 0.625, -0.28125]]


dct_4x4 = [[0.5, 0.5, 0.5, 0.5], [0.65328148, 0.27059805, -0.27059805, -0.65328148], 
           [0.5, -0.5, -0.5, 0.5], [0.27059805, -0.65328148, 0.65328148, -0.27059805]]
dct_4x4_binary = [[0.5, 0.5, 0.5, 0.5], [0.625, 0.25, -0.25, -0.625],
                  [0.5, -0.5, -0.5, 0.5], [0.25, -0.625, 0.625, -0.25]]
result = np.dot(np.transpose(dct_4x4_binary),dct_4x4_binary)
inverse_result = np.linalg.inv(result)
inverse_dct_4x4_binary = np.linalg.inv(dct_4x4_binary)
a = [[0.5, 0.625, ]]
print(f'result:\n{result}\ninverse_result:\n{inverse_result}\ninverse_dct_4x4_binary:\n{inverse_dct_4x4_binary}')