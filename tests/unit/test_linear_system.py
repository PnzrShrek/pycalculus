import pytest
from solvers.linear_system import LinearSystemSolver

class TestLinearSystemSolver:
    def test_validate_input_valid(self):
        """Test input validation with correct dimensions"""
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        solver = LinearSystemSolver(matrix, vector)
        assert solver.validate_input() is True


    def test_solve_2x2_system(self):
        """Test solving a simple 2x2 system"""
        matrix = [[2, 1], [1, 3]]
        vector = [5, 10]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([1.0, 3.0])
        assert result['message'] == 'Solution found'

    def test_solve_3x3_system(self):
        """Test solving a 3x3 system"""
        matrix = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
        vector = [1, -2, 0]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([1.0, -2.0, -2.0])
        assert result['message'] == 'Solution found'

    def test_solve_singular_matrix(self):
        """Test solving with a singular matrix"""
        matrix = [[1, 2], [2, 4]]  # Second row is multiple of first
        vector = [3, 6]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert result['is_singular']
        assert result['solution'] is None
        assert result['message'] == 'Matrix is singular or nearly singular'


    def test_solve_with_pivoting(self):
        """Test that pivoting works correctly"""
        matrix = [[0, 1], [1, 0]]
        vector = [1, 2]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([2.0, 1.0])
        assert result['message'] == 'Solution found'

    def test_solve_identity_matrix(self):
        """Test solving with identity matrix"""
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        vector = [1, 2, 3]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([1.0, 2.0, 3.0])
        assert result['message'] == 'Solution found'

    def test_solve_diagonal_matrix(self):
        """Test solving with diagonal matrix"""
        matrix = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        vector = [4, 9, 16]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([2.0, 3.0, 4.0])
        assert result['message'] == 'Solution found'

    def test_solve_ill_conditioned_system(self):
        """Test solving an ill-conditioned system (Hilbert matrix)"""
        matrix = [[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]]
        vector = [11/6, 13/12, 47/60]  # Solution is [1, 1, 1]
        solver = LinearSystemSolver(matrix, vector)
        result = solver.solve()
        assert not result['is_singular']
        assert result['solution'] == pytest.approx([1.0, 1.0, 1.0], rel=1e-6)
        assert result['message'] == 'Solution found'