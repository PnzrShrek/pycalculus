"""
Numerical interpolation solver.
Implements Lagrange, Newton, and cubic spline interpolation methods.
"""
from typing import List, Tuple, Dict, Callable
from .base import MathSolver


class Interpolator(MathSolver):
    def __init__(self, points: List[Tuple[float, float]], method: str = 'lagrange'):
        """
        Initialize interpolator with data points.

        Args:
            points: List of (x,y) data points
            method: Interpolation method ('lagrange', 'newton', or 'spline')
        """
        self.points = points
        self.method = method

    def validate_input(self) -> bool:
        """Validate input points and method"""
        if len(self.points) < 2:
            raise ValueError("At least 2 points are required for interpolation")
        x_values = [p[0] for p in self.points]
        if len(x_values) != len(set(x_values)):
            raise ValueError("X values must be unique")
        if self.method not in ['lagrange', 'newton', 'spline']:
            raise ValueError("Method must be 'lagrange', 'newton' or 'spline'")
        return True

    def solve(self) -> Dict:
        """Create interpolation function using specified method"""
        self.validate_input()
        if self.method == 'lagrange':
            return self._lagrange_interpolation()
        elif self.method == 'newton':
            return self._newton_interpolation()
        else:
            return self._spline_interpolation()

    def _lagrange_interpolation(self) -> Dict:
        """Lagrange polynomial interpolation"""
        n = len(self.points)

        def lagrange_poly(x: float) -> float:
            result = 0.0
            for i in range(n):
                xi, yi = self.points[i]
                term = yi
                for j in range(n):
                    if i != j:
                        xj, _ = self.points[j]
                        term *= (x - xj) / (xi - xj)
                result += term
            return result

        return {
            'function': lagrange_poly,
            'method': 'Lagrange',
            'degree': n - 1
        }

    def _newton_interpolation(self) -> Dict:
        """Newton divided differences interpolation"""
        n = len(self.points)
        # Initialize divided differences table
        diff_table = [[0.0 for _ in range(n)] for _ in range(n)]

        # Fill first column with y values
        for i in range(n):
            diff_table[i][0] = self.points[i][1]

        # Compute divided differences
        for j in range(1, n):
            for i in range(n - j):
                x_i = self.points[i][0]
                x_ij = self.points[i + j][0]
                diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x_ij - x_i)

        def newton_poly(x: float) -> float:
            """Newton interpolation polynomial"""
            result = diff_table[0][0]
            product = 1.0
            for i in range(1, n):
                product *= (x - self.points[i - 1][0])
                result += diff_table[0][i] * product
            return result

        return {
            'function': newton_poly,
            'method': 'Newton',
            'degree': n - 1
        }

    def _spline_interpolation(self) -> Dict:
        """Cubic spline interpolation"""
        n = len(self.points)
        if n < 3:
            raise ValueError("Spline interpolation requires at least 3 points")

        # Sort points by x value
        points = sorted(self.points, key=lambda p: p[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Calculate differences between x points
        h = [x[i + 1] - x[i] for i in range(n - 1)]

        # Initialize tridiagonal system for spline coefficients
        alpha = [0.0] * (n - 1)
        for i in range(1, n - 1):
            alpha[i] = 3 / h[i] * (y[i + 1] - y[i]) - 3 / h[i - 1] * (y[i] - y[i - 1])

        # Thomas algorithm for tridiagonal system
        l = [1.0] * n
        mu = [0.0] * n
        z = [0.0] * n
        c = [0.0] * n

        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # Back substitution
        for i in reversed(range(n - 1)):
            c[i] = z[i] - mu[i] * c[i + 1]

        # Calculate remaining coefficients
        b = [0.0] * (n - 1)
        d = [0.0] * (n - 1)
        for i in range(n - 1):
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        def spline_function(x_val: float) -> float:
            """Piecewise cubic spline function"""
            if x_val <= x[0]:
                i = 0
            elif x_val >= x[-1]:
                i = n - 2
            else:
                # Find the right interval
                for i in range(n - 1):
                    if x_val <= x[i + 1]:
                        break
            dx = x_val - x[i]
            return y[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

        return {
            'function': spline_function,
            'method': 'Cubic Spline',
            'segments': n - 1
        }
