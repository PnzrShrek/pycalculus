import pytest
from solvers.base import MathSolver


def test_math_solver_abstract_class():
    """Test that MathSolver cannot be instantiated directly."""
    with pytest.raises(TypeError):
        solver = MathSolver()


def test_math_solver_subclass():
    """Test that concrete subclass implements all abstract methods."""

    class ConcreteSolver(MathSolver):
        def solve(self):
            return {"result": "test"}

        def validate_input(self):
            return True

    solver = ConcreteSolver()
    assert solver.solve() == {"result": "test"}
    assert solver.validate_input() is True