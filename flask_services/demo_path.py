"""
    demo_path.py

    USAGE: `import demo_path` in a file's imports.

    This script searches the file tree to find the root
    directory of the project and adds it to python's
    sys.path.

    TROUBLESHOOTING: The `dir_name` argument should be
    repository's root on the system running the script.
    If the git repo has been renamed, then that name
    should be changed here too.
"""

import os
import sys


def get_dir_path(dir_name):
    base_path = os.getcwd().split("/")

    while base_path[-1] != str(dir_name) and base_path:
        base_path = base_path[:-1]

    if not base_path:
        raise ValueError("Base directory name not found. \n"
                         "Did you rename the demo's repository?")
    else:
        return "/".join(base_path) + "/"


base_name = 'p5_afm_2018_demo'
base_dir = get_dir_path(base_name)
sys.path.append(base_dir)

print("Added `" + base_dir + "` to PATH.")
