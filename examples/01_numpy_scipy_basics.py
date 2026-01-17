#!/usr/bin/env python3
"""
NumPy and SciPy Basics
======================
Demonstrates fundamental operations with NumPy and SciPy.

NumPy: https://numpy.org/
SciPy: https://scipy.org/
"""

import numpy as np
from scipy import stats, linalg, optimize

# =============================================================================
# NumPy Array Operations
# =============================================================================
print("=" * 60)
print("NumPy Array Operations")
print("=" * 60)

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random_arr = np.random.randn(5, 5)

print(f"1D Array: {arr}")
print(f"Matrix shape: {matrix.shape}")
print(f"Random array mean: {random_arr.mean():.4f}")

# Array operations
print(f"\nArray sum: {arr.sum()}")
print(f"Array mean: {arr.mean()}")
print(f"Array std: {arr.std():.4f}")

# Broadcasting
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
print(f"\nBroadcasting result:\n{a + b}")

# =============================================================================
# SciPy Statistical Functions
# =============================================================================
print("\n" + "=" * 60)
print("SciPy Statistical Functions")
print("=" * 60)

# Generate sample data
data = np.random.normal(loc=100, scale=15, size=1000)

# Descriptive statistics
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std Dev: {np.std(data):.2f}")
print(f"Skewness: {stats.skew(data):.4f}")
print(f"Kurtosis: {stats.kurtosis(data):.4f}")

# Statistical tests
stat, p_value = stats.normaltest(data)
print(f"\nNormality test p-value: {p_value:.4f}")

# Confidence interval
ci = stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")

# =============================================================================
# SciPy Linear Algebra
# =============================================================================
print("\n" + "=" * 60)
print("SciPy Linear Algebra")
print("=" * 60)

# Matrix operations
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Solve linear system Ax = b
x = linalg.solve(A, b)
print(f"Solution to Ax = b: {x}")

# Eigenvalues
eigenvalues, eigenvectors = linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")

# Matrix decomposition
U, s, Vh = linalg.svd(A)
print(f"Singular values: {s}")

# =============================================================================
# SciPy Optimization
# =============================================================================
print("\n" + "=" * 60)
print("SciPy Optimization")
print("=" * 60)


# Define a function to minimize
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


x0 = np.array([0, 0])
result = optimize.minimize(rosenbrock, x0, method="BFGS")
print(f"Minimum found at: {result.x}")
print(f"Function value at minimum: {result.fun:.6f}")

print("\n" + "=" * 60)
print("NumPy/SciPy example complete!")
print("=" * 60)
