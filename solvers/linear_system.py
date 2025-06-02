"""
Solver for systems of linear equations.
Implements Gaussian elimination with partial pivoting.
"""
from typing import List, Dict
from .base import MathSolver


class LinearSystemSolver(MathSolver):
    def __init__(self, matrix: List[List[float]], vector: List[float]):
        """
        Initialize with coefficient matrix and right-hand side vector.

        Args:
            matrix: Square coefficient matrix
            vector: Right-hand side vector
        """
        self.matrix = matrix
        self.vector = vector
        self.precision = 1e-10  # Threshold for considering value as zero

    def validate_input(self) -> bool:
        """Validate matrix and vector dimensions"""
        n = len(self.matrix)
        if n == 0:
            raise ValueError("Matrix cannot be empty")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("Matrix must be square")
        if len(self.vector) != n:
            raise ValueError("Vector dimension must match matrix size")
        return True

    def solve(self) -> Dict:
        """Solve the linear system using Gaussian elimination"""
        self.validate_input()
        n = len(self.matrix)
        matrix = [row.copy() for row in self.matrix]
        vector = self.vector.copy()

        # Forward elimination with partial pivoting
        for col in range(n):
            # Find row with maximum element in current column
            max_row = col
            for row in range(col + 1, n):
                if abs(matrix[row][col]) > abs(matrix[max_row][col]):
                    max_row = row

            # Swap rows if necessary
            if max_row != col:
                matrix[col], matrix[max_row] = matrix[max_row], matrix[col]
                vector[col], vector[max_row] = vector[max_row], vector[col]

            # Check for singular matrix
            if abs(matrix[col][col]) < self.precision:
                return {
                    'solution': None,
                    'is_singular': True,
                    'message': 'Matrix is singular or nearly singular'
                }

            # Eliminate current column in lower rows
            for row in range(col + 1, n):
                factor = matrix[row][col] / matrix[col][col]
                vector[row] -= factor * vector[col]
                for c in range(col, n):
                    matrix[row][c] -= factor * matrix[col][c]

        # Back substitution
        solution = [0.0 for _ in range(n)]
        for row in reversed(range(n)):
            sum_ax = 0.0
            for col in range(row + 1, n):
                sum_ax += matrix[row][col] * solution[col]
            solution[row] = (vector[row] - sum_ax) / matrix[row][row]

        return {
            'solution': solution,
            'is_singular': False,
            'message': 'Solution found'
        }
