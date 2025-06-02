"""
Solver for ordinary differential equations (ODEs).
Implements Euler and Runge-Kutta 4th order methods.
"""
from typing import List, Tuple, Dict, Callable
from .base import MathSolver


class DifferentialEquationSolver(MathSolver):
    def __init__(self, equation: Callable, method: str = 'rk4'):
        """
        Initialize ODE solver with differential equation.

        Args:
            equation: Function representing dy/dx = f(x,y)
            method: Integration method ('euler' or 'rk4')
        """
        self.equation = equation
        self.method = method
        self.step_size = 0.1  # Default step size

    def validate_input(self) -> bool:
        """Validate equation and method"""
        if not callable(self.equation):
            raise ValueError("Equation must be callable")
        if self.method not in ['euler', 'rk4']:
            raise ValueError("Method must be 'euler' or 'rk4'")
        return True

    def solve(self, x0: float, y0: float, x_end: float) -> Dict:
        """
        Solve ODE with initial conditions over interval.

        Args:
            x0: Initial x value
            y0: Initial y value at x0
            x_end: End of integration interval
        """
        self.validate_input()
        if x_end <= x0:
            raise ValueError("End point must be greater than initial point")

        points = [(x0, y0)]
        x = x0
        y = y0

        while x < x_end:
            h = min(self.step_size, x_end - x)  # Adjust step to not overshoot

            if self.method == 'euler':
                # Euler's method (first order)
                y += h * self.equation(x, y)
            else:
                # Runge-Kutta 4th order method
                k1 = self.equation(x, y)
                k2 = self.equation(x + h / 2, y + h / 2 * k1)
                k3 = self.equation(x + h / 2, y + h / 2 * k2)
                k4 = self.equation(x + h, y + h * k3)
                y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            x += h
            points.append((x, y))

        return {
            'points': points,
            'method': self.method,
            'step_size': self.step_size
        }
