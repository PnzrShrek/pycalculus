import os
import sys
import subprocess
from interfaces.gui import MathAppGUI  # Adjust import to your structure
import tkinter as tk

def run_tox_tests():
    """Run tox tests and display colored output"""
    print("\nRunning tests...\n")
    try:
        # Run tox and capture output
        result = subprocess.run(
            ["tox"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        return False

if __name__ == "__main__":
    # Run tests first
    if not run_tox_tests():
        sys.exit(1)

    # Only proceed if tests pass
    print("Starting main application...")
    root = tk.Tk()
    app = MathAppGUI(root)
    root.mainloop()