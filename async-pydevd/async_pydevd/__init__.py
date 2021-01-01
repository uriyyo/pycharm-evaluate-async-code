from pathlib import Path
from typing import Tuple

import nest_asyncio

ROOT: Path = Path(__file__).parent

FILES: Tuple[Path, ...] = (
    Path(nest_asyncio.__file__),
    ROOT / "asyncio_patch.py",
    ROOT / "async_eval.py",
    ROOT / "pydevd_patch.py",
    ROOT / "pydevd_main.py",
)


def generate() -> str:
    return (
        "\n".join(p.read_text("utf-8") for p in FILES)
        .replace('"""', "'''")
        .replace("  # pragma: no cover", "")
        .strip()
    )


__all__ = ["generate", "FILES"]
