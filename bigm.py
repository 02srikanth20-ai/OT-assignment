import numpy as np

# Big-M Simplex Method

# Minimize:
# Z = 4x1 + x2

# Subject to:
# 3x1 + x2 = 3
# 4x1 + 3x2 >= 6
# x1 + 2x2 <= 4

M = 1000000.0

# Variable order:
# x1, x2, s1, s2, a1, a2, s3

A = np.array([
    [3, 1, 0, 0, 1, 0, 0],
    [4, 3, 0, -1, 0, 1, 0],
    [1, 2, 1, 0, 0, 0, 1]
], dtype=float)

b = np.array([
    3,
    6,
    4
], dtype=float)

# Convert minimization into maximization

c = np.array([
    -4,
    -1,
    0,
    0,
    -M,
    -M,
    0
], dtype=float)

# Initial basis:
# a1, a2, s3

basis = [4, 5, 2]

while True:

    cb = c[basis]

    reduced_cost = c - cb @ A

    # Check optimality

    if max(reduced_cost) <= 1e-9:
        break

    # Entering variable

    entering = int(np.argmax(reduced_cost))

    ratios = []

    for i in range(len(b)):

        if A[i, entering] > 1e-9:

            ratios.append(
                b[i] / A[i, entering]
            )

        else:

            ratios.append(
                float("inf")
            )

    # Leaving variable

    leaving = int(np.argmin(ratios))

    # Pivot operation

    pivot = A[leaving, entering]

    A[leaving] = A[leaving] / pivot

    b[leaving] = b[leaving] / pivot

    for i in range(len(b)):

        if i != leaving:

            factor = A[i, entering]

            A[i] = A[i] - factor * A[leaving]

            b[i] = b[i] - factor * b[leaving]

    basis[leaving] = entering


# Store solution

solution = np.zeros(7)

for i in range(len(basis)):

    solution[basis[i]] = b[i]


max_value = c[basis] @ b

min_value = -max_value


print("Optimal x1 =", round(solution[0], 4))

print("Optimal x2 =", round(solution[1], 4))

print("Minimum Z =", round(min_value, 4))