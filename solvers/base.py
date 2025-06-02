"""
Abstract base class for all mathematical solvers.
Defines the common interface that all solvers must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict


class MathSolver(ABC):
    @abstractmethod
    def solve(self) -> Dict:
        """Main method to solve the mathematical problem"""
        pass

    @abstractmethod
    def validate_input(self) -> bool:
        """Validate input parameters before solving"""
        pass
