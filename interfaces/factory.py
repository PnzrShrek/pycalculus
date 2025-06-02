# In interfaces/factory.py
from solvers.extremum import ExtremumFinder
from solvers.linear_system import LinearSystemSolver
from solvers.differential import DifferentialEquationSolver
from solvers.integral import Integrator
from solvers.interpolation import Interpolator

class MathSolverFactory:
    @staticmethod
    def create_solver(problem_type: str, *args, **kwargs):
        """
        Create appropriate solver instance based on problem type.

        Args:
            problem_type: Type of problem to solve
            *args: Positional arguments for solver constructor
            **kwargs: Keyword arguments for solver constructor

        Returns:
            Instance of appropriate solver class

        Raises:
            ValueError for unknown problem types
        """
        solvers = {
            'extremum': ExtremumFinder,
            'linear_system': LinearSystemSolver,
            'differential': DifferentialEquationSolver,
            'integral': Integrator,
            'interpolation': Interpolator
        }

        if problem_type not in solvers:
            raise ValueError("Unknown problem type")

        return solvers[problem_type](*args, **kwargs)