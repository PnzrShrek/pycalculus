import pytest
import numpy as np
from math import sin, pi, exp
from solvers.integral import Integrator


class TestIntegrator:
    # Test functions
    def linear_func(self, x):
        return 2 * x + 3

    def quadratic_func(self, x):
        return x ** 2 + 2 * x + 1

    def sin_func(self, x):
        return sin(x)

    def exp_func(self, x):
        return exp(-x ** 2)

    # 2D test function
    def func_2d(self, x, y):
        return x ** 2 + y ** 2

    def test_validate_input_valid(self):
        """Test input validation with correct inputs"""
        integrator = Integrator(self.linear_func)
        assert integrator.validate_input() is True


    def test_trapezoid_rule_linear(self):
        """Test trapezoid rule with linear function (should be exact)"""
        integrator = Integrator(self.linear_func, 'trapezoid')
        result = integrator.solve(0, 2)
        exact = 10.0  # ∫(2x+3)dx from 0 to 2 = x²+3x = 4+6 = 10
        assert result['value'] == pytest.approx(exact, rel=1e-6)
        assert result['converged'] is True
        assert result['method'] == 'Trapezoidal Rule'

    def test_simpson_rule_quadratic(self):
        """Test Simpson's rule with quadratic function (should be exact)"""
        integrator = Integrator(self.quadratic_func, 'simpson')
        result = integrator.solve(0, 2)
        exact = 26 / 3  # ∫(x²+2x+1)dx from 0 to 2 = x³/3 + x² + x = 8/3 + 4 + 2
        assert result['value'] == pytest.approx(exact, rel=1e-6)
        assert result['converged'] is True
        assert result['method'] == "Simpson's Rule"

    def test_adaptive_refinement(self):
        """Test that adaptive refinement works"""
        integrator = Integrator(self.exp_func, 'trapezoid')
        integrator.precision = 1e-4
        result = integrator.solve(0, 1)
        # Gaussian integral doesn't have simple exact solution, just check convergence
        assert result['iterations'] > 0
        assert result['converged'] is True
        assert result['segments'] > 2

    def test_multi_integral_invalid_bounds(self):
        """Test invalid bounds for multi-integral"""
        integrator = Integrator(self.func_2d)
        with pytest.raises(ValueError, match="Each dimension must have exactly 2 bounds"):
            integrator.solve(0, 1, (0,))  # Invalid bounds

    def test_default_method(self):
        """Test default method (trapezoid)"""
        integrator = Integrator(self.linear_func)
        result = integrator.solve(0, 2)
        assert result['method'] == 'Trapezoidal Rule'

    def test_result_structure(self):
        """Test that result has correct structure"""
        integrator = Integrator(self.linear_func)
        result = integrator.solve(0, 1)

        assert isinstance(result, dict)
        assert 'value' in result
        assert 'method' in result
        assert 'segments' in result
        assert 'iterations' in result
        assert 'converged' in result
        assert isinstance(result['value'], float)
        assert isinstance(result['segments'], int)
        assert isinstance(result['iterations'], int)
        assert isinstance(result['converged'], bool)

    def test_max_iterations(self):
        """Test that max iterations works"""
        integrator = Integrator(self.sin_func)
        integrator.precision = 1e-12  # Unreachable precision
        integrator.max_iterations = 5
        result = integrator.solve(0, pi)
        assert result['iterations'] == 5
        assert result['converged'] is False