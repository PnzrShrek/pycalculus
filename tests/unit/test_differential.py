import pytest
from math import isclose, exp
from solvers.differential import DifferentialEquationSolver


class TestDifferentialEquationSolver:
    # Test equation: dy/dx = y (solution is y = e^x)
    def exponential_equation(self, x, y):
        return y

    # Test equation: dy/dx = 2x (solution is y = x^2 + C)
    def quadratic_equation(self, x, y):
        return 2 * x

    def test_validate_input_valid(self):
        """Test input validation with correct inputs"""
        solver = DifferentialEquationSolver(self.exponential_equation, 'rk4')
        assert solver.validate_input() is True

    def test_solve_invalid_interval(self):
        """Test solving with invalid interval"""
        solver = DifferentialEquationSolver(self.exponential_equation)
        with pytest.raises(ValueError, match="End point must be greater than initial point"):
            solver.solve(0, 1, -1)  # x_end < x0

    def test_solve_euler_exponential(self):
        """Test Euler's method with exponential equation"""
        solver = DifferentialEquationSolver(self.exponential_equation, 'euler')
        solver.step_size = 0.01  # Small step for better accuracy
        result = solver.solve(0, 1, 1)  # y = e^x from 0 to 1

        # Check final value against exact solution (e^1 ≈ 2.71828)
        final_x, final_y = result['points'][-1]
        assert isclose(final_x, 1.0)
        assert isclose(final_y, exp(1), rel_tol=1e-2)  # Euler has low accuracy

    def test_solve_rk4_exponential(self):
        """Test RK4 method with exponential equation"""
        solver = DifferentialEquationSolver(self.exponential_equation, 'rk4')
        solver.step_size = 0.1
        result = solver.solve(0, 1, 1)  # y = e^x from 0 to 1

        # Check final value against exact solution
        final_x, final_y = result['points'][-1]
        assert isclose(final_x, 1.0)
        assert isclose(final_y, exp(1), rel_tol=1e-6)  # RK4 should be very accurate

    def test_solve_euler_quadratic(self):
        """Test Euler's method with quadratic equation"""
        solver = DifferentialEquationSolver(self.quadratic_equation, 'euler')
        solver.step_size = 0.01
        result = solver.solve(0, 0, 2)  # y = x^2 from 0 to 2 (initial y=0)

        final_x, final_y = result['points'][-1]
        assert isclose(final_x, 2.0)
        assert isclose(final_y, 4.0, rel_tol=1e-2)  # Exact solution is 4

    def test_solve_rk4_quadratic(self):
        """Test RK4 method with quadratic equation"""
        solver = DifferentialEquationSolver(self.quadratic_equation, 'rk4')
        solver.step_size = 0.1
        result = solver.solve(0, 0, 2)  # y = x^2 from 0 to 2

        final_x, final_y = result['points'][-1]
        assert isclose(final_x, 2.0)
        assert isclose(final_y, 4.0, rel_tol=1e-6)  # RK4 should be exact for quadratic

    def test_result_structure(self):
        """Test that result has correct structure"""
        solver = DifferentialEquationSolver(self.exponential_equation)
        result = solver.solve(0, 1, 0.5)

        assert isinstance(result, dict)
        assert 'points' in result
        assert 'method' in result
        assert 'step_size' in result
        assert isinstance(result['points'], list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result['points'])

    def test_step_size_adjustment(self):
        """Test that step size is adjusted to not overshoot x_end"""
        solver = DifferentialEquationSolver(self.exponential_equation)
        solver.step_size = 0.3
        result = solver.solve(0, 1, 1.0)

        # Check that we exactly reach x_end
        final_x = result['points'][-1][0]
        assert isclose(final_x, 1.0)

        # Check that steps are correct (0.3, 0.3, 0.3, 0.1)
        x_values = [p[0] for p in result['points']]
        assert len(x_values) == 5  # Initial + 4 steps
        assert isclose(x_values[1] - x_values[0], 0.3)
        assert isclose(x_values[-1] - x_values[-2], 0.1)

    def test_compare_methods_accuracy(self):
        """Compare accuracy of Euler vs RK4 methods"""
        # Use same step size for both
        step = 0.1
        exact_solution = exp(1)

        # Euler solution
        euler_solver = DifferentialEquationSolver(self.exponential_equation, 'euler')
        euler_solver.step_size = step
        euler_result = euler_solver.solve(0, 1, 1)
        euler_error = abs(euler_result['points'][-1][1] - exact_solution)

        # RK4 solution
        rk4_solver = DifferentialEquationSolver(self.exponential_equation, 'rk4')
        rk4_solver.step_size = step
        rk4_result = rk4_solver.solve(0, 1, 1)
        rk4_error = abs(rk4_result['points'][-1][1] - exact_solution)

        # RK4 should be significantly more accurate
        assert rk4_error < euler_error / 100  # RK4 error should be much smaller