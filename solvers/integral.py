"""
Numerical integration solver.
Implements trapezoid rule, Simpson's rule, and Monte Carlo methods.
"""
import numpy as np
from typing import List, Tuple, Dict, Callable
from .base import MathSolver


class Integrator(MathSolver):
    def __init__(self, func: Callable, method: str = 'trapezoid'):
        """
        Initialize integrator with function to integrate.

        Args:
            func: Function to integrate
            method: Integration method ('trapezoid', 'simpson', or 'monte_carlo')
        """
        self.func = func
        self.method = method
        self.precision = 1e-6  # Convergence threshold
        self.max_iterations = 1000  # Maximum refinement iterations

    def validate_input(self) -> bool:
        """Validate function and method"""
        if not callable(self.func):
            raise ValueError("Function must be callable")
        return True

    def solve(self, a: float, b: float, *args) -> Dict:
        """
        Compute integral of function from a to b.
        Supports multidimensional integrals through repeated 1D integration.
        """
        self.validate_input()
        if len(args) == 0:
            return self._single_integral(a, b)
        else:
            return self._multi_integral(a, b, *args)

    def _single_integral(self, a: float, b: float) -> Dict:
        """Вычисление одномерного интеграла с указанием метода"""
        if self.method == 'trapezoid':
            result = self._trapezoid_rule(a, b)
            result['method'] = 'Trapezoidal Rule'
        elif self.method == 'simpson':
            result = self._simpson_rule(a, b)
            result['method'] = "Simpson's Rule"
        else:
            result = self._monte_carlo_1d(a, b)
            result['method'] = 'Monte Carlo'
        return result

    def _multi_integral(self, *bounds) -> Dict:
        """Compute multidimensional integral through repeated 1D integration"""
        result = 1.0
        current_bounds = []
        for bound in bounds:
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                raise ValueError("Each dimension must have exactly 2 bounds")
            a, b = bound
            current_bounds.append((a, b))

            def partial_func(*args):
                return self.func(*args)

            integral = self._single_integral(a, b)
            result *= integral['value']
        return {
            'value': result,
            'method': f"Repeated 1D {self.method}",
            'bounds': current_bounds
        }

    def _trapezoid_rule(self, a: float, b: float) -> Dict:
        """Trapezoidal rule with adaptive refinement"""
        n = 2  # Start with 2 segments
        prev_integral = 0.0
        iterations = 0

        while iterations < self.max_iterations:
            h = (b - a) / n
            integral = 0.5 * (self.func(a) + self.func(b))

            # Sum function values at intermediate points
            for i in range(1, n):
                integral += self.func(a + i * h)
            integral *= h

            # Check for convergence
            if iterations > 0 and abs(integral - prev_integral) < self.precision:
                break

            prev_integral = integral
            n *= 2  # Double number of segments for next iteration
            iterations += 1

        return {
            'value': integral,
            'segments': n,
            'iterations': iterations,
            'converged': iterations < self.max_iterations
        }

    def _simpson_rule(self, a: float, b: float) -> Dict:
        """Simpson's rule with adaptive refinement"""
        n = 2  # Start with 2 segments (must be even)
        prev_integral = 0.0
        iterations = 0

        while iterations < self.max_iterations:
            h = (b - a) / n
            integral = self.func(a) + self.func(b)

            # Weighted sum of function values
            for i in range(1, n):
                x = a + i * h
                if i % 2 == 1:  # Odd indices get weight 4
                    integral += 4 * self.func(x)
                else:  # Even indices get weight 2
                    integral += 2 * self.func(x)
            integral *= h / 3

            # Check for convergence
            if iterations > 0 and abs(integral - prev_integral) < self.precision:
                break

            prev_integral = integral
            n *= 2  # Double number of segments
            iterations += 1

        return {
            'value': integral,
            'segments': n,
            'iterations': iterations,
            'converged': iterations < self.max_iterations
        }

    import numpy as np

    def _monte_carlo_1d(self, a: float, b: float) -> Dict:
        """Monte Carlo integration for 1D functions using NumPy"""
        # Initial parameters
        n = 3  # Initial number of samples
        prev_integral = 0.0
        iterations = 0
        range_width = b - a

        # Estimate maximum function value more accurately
        x_test = np.linspace(a, b, 21)  # 21 points for better max estimation
        max_f = np.max(self.func(x_test))

        while iterations < self.max_iterations:
            # Generate all random samples at once
            x_samples = np.random.uniform(a, b, n)
            y_samples = np.random.uniform(0, max_f, n)

            # Vectorized function evaluation and comparison
            hits = np.sum(y_samples <= self.func(x_samples))
            integral = (hits / n) * range_width * max_f

            # Check convergence
            if iterations > 0 and abs(integral - prev_integral) < self.precision:
                break

            prev_integral = integral
            n *= 2  # Double sample count for next iteration
            iterations += 1

        return {
            'value': integral,
            'segments': n,
            'iterations': iterations,
            'converged': iterations < self.max_iterations
        }
