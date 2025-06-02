import pytest
import math
from solvers.extremum import ExtremumFinder


@pytest.fixture
def quadratic_func():
    return lambda x, y: x ** 2 + y ** 2

def test_extremum_finder_validation(quadratic_func):
    solver = ExtremumFinder(quadratic_func, ['x', 'y'])
    assert solver.validate_input() is True

    solver = ExtremumFinder("not a function", ['x'])
    with pytest.raises(ValueError, match="Function must be callable"):
        solver.validate_input()


def test_gradient_descent(quadratic_func):
    solver = ExtremumFinder(quadratic_func, ['x', 'y'])
    result = solver.solve([1.0, 1.0])
    assert result['converged'] is True
    assert math.isclose(result['point'][0], 0, abs_tol=1e-4)
    assert math.isclose(result['point'][1], 0, abs_tol=1e-4)
    assert math.isclose(result['value'], 0, abs_tol=1e-4)


def test_newton_method(quadratic_func):
    solver = ExtremumFinder(quadratic_func, ['x', 'y'], method='newton')
    result = solver.solve([1.0, 1.0])
    assert result['converged'] is True
    assert math.isclose(result['point'][0], 0, abs_tol=1e-4)
    assert math.isclose(result['point'][1], 0, abs_tol=1e-4)


def test_compute_gradient(quadratic_func):
    solver = ExtremumFinder(quadratic_func, ['x', 'y'])
    grad = solver._compute_gradient([1.0, 2.0])
    assert len(grad) == 2
    assert math.isclose(grad[0], 2.0, abs_tol=1e-4)
    assert math.isclose(grad[1], 4.0, abs_tol=1e-4)


def test_compute_hessian(quadratic_func):
    solver = ExtremumFinder(quadratic_func, ['x', 'y'])
    hessian = solver._compute_hessian([1.0, 2.0])
    assert len(hessian) == 2
    assert len(hessian[0]) == 2
    assert math.isclose(hessian[0][0], 2.0, abs_tol=1e-2)
    assert math.isclose(hessian[1][1], 2.0, abs_tol=1e-2)
    assert math.isclose(hessian[0][1], 0.0, abs_tol=1e-2)