import pytest
from solvers.interpolation import Interpolator


@pytest.fixture
def sample_points():
    return [(0, 0), (1, 1), (2, 4)]


def test_interpolator_validation(sample_points):
    solver = Interpolator(sample_points)
    assert solver.validate_input() is True

    # Test insufficient points
    with pytest.raises(ValueError, match="At least 2 points are required"):
        Interpolator([(0, 0)]).validate_input()

    # Test duplicate x values
    with pytest.raises(ValueError, match="X values must be unique"):
        Interpolator([(0, 0), (0, 1)]).validate_input()

    # Test invalid method
    with pytest.raises(ValueError, match="Method must be 'lagrange', 'newton' or 'spline'"):
        Interpolator(sample_points, method='invalid').validate_input()


def test_lagrange_interpolation(sample_points):
    solver = Interpolator(sample_points, method='lagrange')
    result = solver.solve()
    assert callable(result['function'])
    assert result['method'] == 'Lagrange'
    assert result['degree'] == 2
    assert result['function'](1.5) == pytest.approx(2.25)


def test_newton_interpolation(sample_points):
    solver = Interpolator(sample_points, method='newton')
    result = solver.solve()
    assert callable(result['function'])
    assert result['method'] == 'Newton'
    assert result['degree'] == 2
    assert result['function'](1.5) == pytest.approx(2.25)


def test_spline_interpolation(sample_points):
    solver = Interpolator(sample_points, method='spline')
    result = solver.solve()
    assert callable(result['function'])
    assert result['method'] == 'Cubic Spline'
    assert result['segments'] == 2
    assert result['function'](1.5) == pytest.approx(2.3125)