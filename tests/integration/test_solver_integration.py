import pytest
from interfaces.factory import MathSolverFactory

def test_extremum_solver_integration():
    solver = MathSolverFactory.create_solver(
        'extremum', lambda x: x**2, ['x'], method='gradient'
    )
    result = solver.solve([1.0])
    assert result['converged'] is True
    assert abs(result['point'][0]) < 1e-4

def test_linear_system_integration():
    solver = MathSolverFactory.create_solver(
        'linear_system', [[2, 1], [1, 3]], [4, 5]
    )
    result = solver.solve()
    assert result['is_singular'] is False
    assert abs(result['solution'][0] - 1.4) < 1e-6

def test_differential_solver_integration():
    solver = MathSolverFactory.create_solver(
        'differential', lambda x, y: x + y, method='rk4'
    )
    result = solver.solve(0, 1, 1)
    assert len(result['points']) > 1
    assert result['points'][-1][0] == 1.0

def test_integral_solver_integration():
    solver = MathSolverFactory.create_solver(
        'integral', lambda x: x**2, method='simpson'
    )
    result = solver.solve(0, 1)
    assert abs(result['value'] - 1/3) < 1e-6

def test_interpolation_solver_integration():
    solver = MathSolverFactory.create_solver(
        'interpolation', [(0, 0), (1, 1), (2, 4)], method='lagrange'
    )
    result = solver.solve()
    assert abs(result['function'](1.5) - 2.25) < 1e-6