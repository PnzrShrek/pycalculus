"""
Solver for finding extrema (minima/maxima) of multivariable functions.
Implements gradient descent and Newton's method for optimization.
"""
import math
from typing import List, Dict, Callable
from .base import MathSolver


class ExtremumFinder(MathSolver):
    def __init__(self, func: Callable, variables: List[str], method: str = 'gradient'):
        """
        Initialize extremum finder with target function and optimization method.

        Args:
            func: The function to optimize
            variables: List of variable names in the function
            method: Optimization method ('gradient' or 'newton')
        """
        self.func = func
        self.variables = variables
        self.method = method
        self.precision = 1e-6  # Convergence threshold
        self.max_iterations = 1000  # Maximum iterations before stopping

    def validate_input(self) -> bool:
        """Validate that inputs are properly formatted"""
        if not callable(self.func):
            raise ValueError("Function must be callable")
        if not isinstance(self.variables, list) or len(self.variables) == 0:
            raise ValueError("Variables must be a non-empty list")
        if self.method not in ['gradient', 'newton']:
            raise ValueError("Method must be 'gradient' or 'newton'")
        return True

    def solve(self, start_point: List[float]) -> Dict:
        """Find extremum starting from given point"""
        self.validate_input()
        if len(start_point) != len(self.variables):
            raise ValueError("Start point dimension must match variables count")

        if self.method == 'gradient':
            return self._gradient_descent(start_point)
        else:
            return self._newton_method(start_point)

    def _gradient_descent(self, start_point: List[float]) -> Dict:
        """Gradient descent optimization implementation"""
        x = start_point.copy()
        step_size = 0.01
        iteration = 0

        while iteration < self.max_iterations:
            grad = self._compute_gradient(x)
            if all(abs(g) < self.precision for g in grad):
                break
            for i in range(len(x)):
                x[i] -= step_size * grad[i]
            iteration += 1

        return {
            'point': x,
            'value': self.func(*x),
            'iterations': iteration,
            'converged': iteration < self.max_iterations
        }

    def _newton_method(self, start_point: List[float]) -> Dict:
        """Newton's method optimization implementation"""
        x = start_point.copy()
        iteration = 0

        while iteration < self.max_iterations:
            grad = self._compute_gradient(x)
            hessian = self._compute_hessian(x)
            if all(abs(g) < self.precision for g in grad):
                break
            step_size = 0.1
            for i in range(len(x)):
                x[i] -= step_size * grad[i] / (hessian[i][i] + 1e-8)
            iteration += 1

        return {
            'point': x,
            'value': self.func(*x),
            'iterations': iteration,
            'converged': iteration < self.max_iterations
        }

    def _compute_gradient(self, point: List[float]) -> List[float]:
        """Compute gradient using central differences"""
        h = 1e-6
        grad = []
        for i in range(len(point)):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h
            partial_derivative = (self.func(*point_plus) - self.func(*point_minus)) / (2 * h)
            grad.append(partial_derivative)
        return grad

    def _compute_hessian(self, point: List[float]) -> List[List[float]]:
        """Compute Hessian matrix using finite differences"""
        h = 1e-6
        n = len(point)
        hessian = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements (second derivatives)
                    point_plus = point.copy()
                    point_minus = point.copy()
                    point_plus[i] += h
                    point_minus[i] -= h
                    f_plus = self.func(*point_plus)
                    f_minus = self.func(*point_minus)
                    f = self.func(*point)
                    hessian[i][j] = (f_plus - 2 * f + f_minus) / (h * h)
                else:
                    # Off-diagonal elements (mixed derivatives)
                    point_pp = point.copy()
                    point_pm = point.copy()
                    point_mp = point.copy()
                    point_mm = point.copy()
                    point_pp[i] += h
                    point_pp[j] += h
                    point_pm[i] += h
                    point_pm[j] -= h
                    point_mp[i] -= h
                    point_mp[j] += h
                    point_mm[i] -= h
                    point_mm[j] -= h
                    hessian[i][j] = (self.func(*point_pp) - self.func(*point_pm) -
                                     self.func(*point_mp) + self.func(*point_mm)) / (4 * h * h)
        return hessian
