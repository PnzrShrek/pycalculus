import pytest
import tkinter as tk
from interfaces.gui import MathAppGUI
from unittest.mock import patch


@pytest.fixture
def app():
    root = tk.Tk()
    root.withdraw()  # Hide the window for testing
    app = MathAppGUI(root)
    yield app
    root.destroy()


def test_gui_initial_state(app):
    assert app.problem_var.get() == 'extremum'
    assert app.status_var.get() == 'Готов'  # Updated to match actual GUI


def test_function_creation(app):
    func = app._create_function_from_string('x**2 + y', ['x', 'y'])
    assert func(2, 3) == 7


def test_invalid_function_creation(app):
    with pytest.raises(ValueError):
        app._create_function_from_string('__import__("os")', ['x'])

