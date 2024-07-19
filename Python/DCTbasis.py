import math

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

# Generate an 8x8 DCT basis matrix
dct_basis_8x8 = generate_dct_basis(8)

print("8x8 DCT Basis Matrix:")
for row in dct_basis_8x8:
    print(row)
