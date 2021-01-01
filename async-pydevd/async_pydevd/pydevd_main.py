import sys
from runpy import run_path

run_path(sys.argv.pop(1), {}, "__main__")
