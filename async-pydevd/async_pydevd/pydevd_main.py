import sys  # pragma: no cover
from runpy import run_path  # pragma: no cover

run_path(sys.argv.pop(1), {}, "__main__")  # pragma: no cover
