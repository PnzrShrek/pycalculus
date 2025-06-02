"""
Graphical user interface for PyCalculus.
Provides interactive access to all mathematical solvers through a Tkinter GUI.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import math
from typing import List, Callable
from solvers import extremum
from solvers import linear_system
from solvers import differential
from solvers import integral
from solvers import interpolation


class MathAppGUI:
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("PyCalculus")
        self.root.geometry("800x600")
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        """Create all GUI widgets"""
        self.main_frame = ttk.Frame(self.root, padding="10")

        # Problem type selection
        self.problem_label = ttk.Label(self.main_frame, text="Тип задачи:")
        self.problem_var = tk.StringVar(value="extremum")
        self.problem_menu = ttk.OptionMenu(
            self.main_frame, self.problem_var, "extremum",
            *["extremum", "linear_system", "differential", "integral", "interpolation"],
            command=self._update_input_fields
        )

        # Method selection
        self.method_label = ttk.Label(self.main_frame, text="Метод:")
        self.method_var = tk.StringVar()
        self.method_menu = ttk.OptionMenu(
            self.main_frame, self.method_var, ""
        )

        # Dynamic input fields frame
        self.input_frame = ttk.Frame(self.main_frame)

        # Solution display
        self.solution_label = ttk.Label(self.main_frame, text="Решение:")
        self.solution_text = tk.Text(
            self.main_frame, height=10, width=60, state=tk.DISABLED
        )

        # Action buttons
        self.solve_button = ttk.Button(
            self.main_frame, text="Решить", command=self._solve_problem
        )
        self.clear_button = ttk.Button(
            self.main_frame, text="Очистить", command=self._clear_fields
        )

        # Status bar
        self.status_var = tk.StringVar(value="Готов")
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )

    def _layout_widgets(self):
        """Arrange widgets in the window"""
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Problem type and method selection
        self.problem_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        self.problem_menu.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.method_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.method_menu.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Dynamic input fields
        self.input_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.W)

        # Solution display
        self.solution_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        self.solution_text.grid(row=4, column=0, columnspan=2, pady=5)

        # Buttons
        self.solve_button.grid(row=5, column=0, pady=10)
        self.clear_button.grid(row=5, column=1, pady=10)

        # Status bar
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize input fields for default problem type
        self._update_input_fields()

    def _update_input_fields(self, *args):
        """Update input fields based on selected problem type"""
        # Clear current input fields
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        problem_type = self.problem_var.get()

        # Update available methods
        menu = self.method_menu["menu"]
        menu.delete(0, "end")

        if problem_type == "extremum":
            methods = ["gradient", "newton"]
        elif problem_type == "linear_system":
            methods = ["gaussian"]
        elif problem_type == "differential":
            methods = ["euler", "rk4"]
        elif problem_type == "integral":
            methods = ["trapezoid", "simpson", "monte_carlo"]
        elif problem_type == "interpolation":
            methods = ["lagrange", "newton", "spline"]

        # Add methods to dropdown menu
        for method in methods:
            menu.add_command(
                label=method,
                command=lambda m=method: self.method_var.set(m)
            )
        self.method_var.set(methods[0])

        # Create appropriate input fields
        if problem_type == "extremum":
            self._create_extremum_inputs()
        elif problem_type == "linear_system":
            self._create_linear_system_inputs()
        elif problem_type == "differential":
            self._create_differential_inputs()
        elif problem_type == "integral":
            self._create_integral_inputs()
        elif problem_type == "interpolation":
            self._create_interpolation_inputs()

    def _create_extremum_inputs(self):
        """Create input fields for extremum finding"""
        ttk.Label(self.input_frame, text="Функция (синтаксис Python):").grid(row=0, column=0, sticky=tk.W)
        self.function_entry = ttk.Entry(self.input_frame, width=40)
        self.function_entry.grid(row=0, column=1, sticky=tk.W)
        self.function_entry.insert(0, "x**2 + 3*x - 2")

        ttk.Label(self.input_frame, text="Переменные (через запятую):").grid(row=1, column=0, sticky=tk.W)
        self.variables_entry = ttk.Entry(self.input_frame, width=40)
        self.variables_entry.grid(row=1, column=1, sticky=tk.W)
        self.variables_entry.insert(0, "x,y")

        ttk.Label(self.input_frame, text="Начальное приближение (через запятую):").grid(row=2, column=0, sticky=tk.W)
        self.initial_guess_entry = ttk.Entry(self.input_frame, width=40)
        self.initial_guess_entry.grid(row=2, column=1, sticky=tk.W)
        self.initial_guess_entry.insert(0, "0,0")

    def _create_linear_system_inputs(self):
        """Create input fields for linear systems"""
        ttk.Label(self.input_frame, text="Матрица (одна строка на линию):").grid(row=0, column=0, sticky=tk.W)
        self.matrix_text = tk.Text(self.input_frame, width=30, height=5)
        self.matrix_text.grid(row=0, column=1, sticky=tk.W)
        self.matrix_text.insert(tk.END, "2 1\n1 3")

        ttk.Label(self.input_frame, text="Вектор (один элемент на линию):").grid(row=1, column=0, sticky=tk.W)
        self.vector_text = tk.Text(self.input_frame, width=30, height=5)
        self.vector_text.grid(row=1, column=1, sticky=tk.W)
        self.vector_text.insert(tk.END, "4\n5")

    def _create_differential_inputs(self):
        """Create input fields for differential equations"""
        ttk.Label(self.input_frame, text="Уравнение (dy/dx = f(x,y)):").grid(row=0, column=0, sticky=tk.W)
        self.equation_entry = ttk.Entry(self.input_frame, width=40)
        self.equation_entry.grid(row=0, column=1, sticky=tk.W)
        self.equation_entry.insert(0, "x + y")

        ttk.Label(self.input_frame, text="Начальное x:").grid(row=1, column=0, sticky=tk.W)
        self.x0_entry = ttk.Entry(self.input_frame, width=10)
        self.x0_entry.grid(row=1, column=1, sticky=tk.W)
        self.x0_entry.insert(0, "0")

        ttk.Label(self.input_frame, text="Начальное y:").grid(row=2, column=0, sticky=tk.W)
        self.y0_entry = ttk.Entry(self.input_frame, width=10)
        self.y0_entry.grid(row=2, column=1, sticky=tk.W)
        self.y0_entry.insert(0, "1")

        ttk.Label(self.input_frame, text="Конечное x:").grid(row=3, column=0, sticky=tk.W)
        self.x_end_entry = ttk.Entry(self.input_frame, width=10)
        self.x_end_entry.grid(row=3, column=1, sticky=tk.W)
        self.x_end_entry.insert(0, "1")

    def _create_integral_inputs(self):
        """Create input fields for integration"""
        ttk.Label(self.input_frame, text="Функция (синтаксис Python):").grid(row=0, column=0, sticky=tk.W)
        self.integral_func_entry = ttk.Entry(self.input_frame, width=40)
        self.integral_func_entry.grid(row=0, column=1, sticky=tk.W)
        self.integral_func_entry.insert(0, "x**2")

        ttk.Label(self.input_frame, text="Нижний предел:").grid(row=1, column=0, sticky=tk.W)
        self.lower_bound_entry = ttk.Entry(self.input_frame, width=10)
        self.lower_bound_entry.grid(row=1, column=1, sticky=tk.W)
        self.lower_bound_entry.insert(0, "0")

        ttk.Label(self.input_frame, text="Верхний предел:").grid(row=2, column=0, sticky=tk.W)
        self.upper_bound_entry = ttk.Entry(self.input_frame, width=10)
        self.upper_bound_entry.grid(row=2, column=1, sticky=tk.W)
        self.upper_bound_entry.insert(0, "1")

    def _create_interpolation_inputs(self):
        """Create input fields for interpolation"""
        ttk.Label(self.input_frame, text="Точки (x,y; одна на линию):").grid(row=0, column=0, sticky=tk.W)
        self.points_text = tk.Text(self.input_frame, width=30, height=5)
        self.points_text.grid(row=0, column=1, sticky=tk.W)
        self.points_text.insert(tk.END, "0 0\n1 1\n2 4")

    def _solve_problem(self):
        """Solve the current problem based on inputs"""
        problem_type = self.problem_var.get()
        method = self.method_var.get()
        try:
            if problem_type == "extremum":
                self._solve_extremum()
            elif problem_type == "linear_system":
                self._solve_linear_system()
            elif problem_type == "differential":
                self._solve_differential()
            elif problem_type == "integral":
                self._solve_integral()
            elif problem_type == "interpolation":
                self._solve_interpolation()

            self.status_var.set("Решение вычислено успешно")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.status_var.set("Ошибка: " + str(e))

    def _solve_extremum(self):
        """Solve extremum finding problem"""
        func_str = self.function_entry.get()
        variables_str = self.variables_entry.get()
        initial_guess_str = self.initial_guess_entry.get()

        variables = [v.strip() for v in variables_str.split(",")]
        func = self._create_function_from_string(func_str, variables)
        initial_guess = [float(x.strip()) for x in initial_guess_str.split(",")]

        solver = extremum.ExtremumFinder(func, variables, self.method_var.get())
        result = solver.solve(initial_guess)

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)

        if result['converged']:
            self.solution_text.insert(tk.END, f"Решение найдено в точке:\n")
            for var, val in zip(variables, result['point']):
                self.solution_text.insert(tk.END, f"{var} = {val:.6f}\n")
            self.solution_text.insert(tk.END, f"\nЗначение функции: {result['value']:.6f}\n")
            self.solution_text.insert(tk.END, f"Итераций: {result['iterations']}\n")
        else:
            self.solution_text.insert(tk.END, "Решение не сошлось\n")
            self.solution_text.insert(tk.END, f"Последняя точка: {result['point']}\n")

        self.solution_text.config(state=tk.DISABLED)

    def _solve_linear_system(self):
        """Solve linear system problem"""
        matrix_lines = self.matrix_text.get("1.0", tk.END).strip().split("\n")
        matrix = []
        for line in matrix_lines:
            row = [float(x) for x in line.split()]
            matrix.append(row)

        vector_lines = self.vector_text.get("1.0", tk.END).strip().split("\n")
        vector = [float(x) for x in vector_lines]

        solver = linear_system.LinearSystemSolver(matrix, vector)
        result = solver.solve()

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)

        if result['is_singular']:
            self.solution_text.insert(tk.END, "Матрица вырождена или почти вырождена\n")
        else:
            self.solution_text.insert(tk.END, "Решение:\n")
            for i, val in enumerate(result['solution']):
                self.solution_text.insert(tk.END, f"x{i} = {val:.6f}\n")

        self.solution_text.config(state=tk.DISABLED)

    def _solve_differential(self):
        """Solve differential equation problem"""
        equation_str = self.equation_entry.get()
        x0 = float(self.x0_entry.get())
        y0 = float(self.y0_entry.get())
        x_end = float(self.x_end_entry.get())

        func = self._create_function_from_string(equation_str, ['x', 'y'])
        solver = differential.DifferentialEquationSolver(func, self.method_var.get())
        result = solver.solve(x0, y0, x_end)

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.insert(tk.END, f"Точки решения (x, y):\n")
        for x, y in result['points']:
            self.solution_text.insert(tk.END, f"{x:.4f}, {y:.6f}\n")
        self.solution_text.insert(tk.END, f"\nМетод: {result['method']}\n")
        self.solution_text.insert(tk.END, f"Шаг: {result['step_size']}\n")
        self.solution_text.config(state=tk.DISABLED)

    def _solve_integral(self):
        """Solve integration problem"""
        func_str = self.integral_func_entry.get()
        a = float(self.lower_bound_entry.get())
        b = float(self.upper_bound_entry.get())

        func = self._create_function_from_string(func_str, ['x'])
        solver = integral.Integrator(func, self.method_var.get())
        result = solver.solve(a, b)

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.insert(tk.END, f"Значение интеграла: {result['value']:.6f}\n")
        if 'segments' in result:
            self.solution_text.insert(tk.END, f"Сегментов: {result['segments']}\n")
        if 'iterations' in result:
            self.solution_text.insert(tk.END, f"Итераций: {result['iterations']}\n")
        self.solution_text.insert(tk.END, f"Метод: {result['method']}\n")
        self.solution_text.config(state=tk.DISABLED)

    def _solve_interpolation(self):
        """Solve interpolation problem"""
        points_lines = self.points_text.get("1.0", tk.END).strip().split("\n")
        points = []
        for line in points_lines:
            x, y = map(float, line.split())
            points.append((x, y))

        solver = interpolation.Interpolator(points, self.method_var.get())
        result = solver.solve()

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.insert(tk.END, f"Интерполяционная функция создана\n")
        self.solution_text.insert(tk.END, f"Метод: {result['method']}\n")
        if 'degree' in result:
            self.solution_text.insert(tk.END, f"Степень: {result['degree']}\n")
        if 'segments' in result:
            self.solution_text.insert(tk.END, f"Сегментов: {result['segments']}\n")

        # Show example evaluation
        example_x = sum(p[0] for p in points) / len(points)
        example_y = result['function'](example_x)
        self.solution_text.insert(tk.END, f"\nПример вычисления при x={example_x:.2f}: {example_y:.6f}\n")
        self.solution_text.config(state=tk.DISABLED)

    def _create_function_from_string(self, func_str: str, variables: List[str]) -> Callable:
        """
        Safely create a function from string input.

        Args:
            func_str: Function expression as string
            variables: List of variable names in the function

        Returns:
            Callable function

        Raises:
            ValueError if function creation fails
        """
        try:
            # Create safe environment for eval
            math_dict = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
            math_dict.update({'__builtins__': None})

            # Add variables to environment
            for var in variables:
                math_dict[var] = None

            # Compile and check for disallowed names
            code = compile(func_str, '<string>', 'eval')
            for name in code.co_names:
                if name not in math_dict:
                    raise ValueError(f"Недопустимое имя в функции: {name}")

            # Create function with variables bound to arguments
            def func(*args):
                local_dict = math_dict.copy()
                for var, arg in zip(variables, args):
                    local_dict[var] = arg
                return eval(code, {'__builtins__': None}, local_dict)

            return func
        except Exception as e:
            raise ValueError(f"Ошибка создания функции: {str(e)}")

    def _clear_fields(self):
        """Clear solution text and reset status"""
        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.config(state=tk.DISABLED)
        self.status_var.set("Готов")
