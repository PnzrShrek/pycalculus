import pytest
from interfaces.factory import MathSolverFactory


def test_factory_creation():
    # Extremum finder
    extremum_solver = MathSolverFactory.create_solver(
        'extremum', lambda x: x ** 2, ['x']
    )
    assert extremum_solver.__class__.__name__ == 'ExtremumFinder'

    # Linear system solver
    linear_solver = MathSolverFactory.create_solver(
        'linear_system', [[1, 0], [0, 1]], [1, 1]
    )
    assert linear_solver.__class__.__name__ == 'LinearSystemSolver'

    # Differential equation solver
    diff_solver = MathSolverFactory.create_solver(
        'differential', lambda x, y: x + y
    )
    assert diff_solver.__class__.__name__ == 'DifferentialEquationSolver'

    # Integral solver
    integral_solver = MathSolverFactory.create_solver(
        'integral', lambda x: x ** 2
    )
    assert integral_solver.__class__.__name__ == 'Integrator'

    # Interpolation solver
    interp_solver = MathSolverFactory.create_solver(
        'interpolation', [(0, 0), (1, 1)]
    )
    assert interp_solver.__class__.__name__ == 'Interpolator'


def test_invalid_solver_type():
    with pytest.raises(ValueError, match="Unknown problem type"):
        MathSolverFactory.create_solver('invalid_type')