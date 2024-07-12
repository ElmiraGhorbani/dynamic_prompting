# import os


# def get_project_root():
#     # Get the absolute path of the current file
#     current_path = os.path.abspath(__file__)
#     # Return the parent directory (project root)
#     return os.path.dirname(os.path.dirname(current_path))

from pathlib import Path


def get_project_root():
    # Get the absolute path of the current file
    current_path = Path(__file__).resolve()
    # Return the parent directory (project root)
    return current_path.parent.parent.parent