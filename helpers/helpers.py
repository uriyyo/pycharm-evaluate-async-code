import re
import subprocess
import textwrap
from pathlib import Path
from typing import Pattern

import click
from async_pydevd import generate

PLUGIN_CHECK_CODE_REGEX: Pattern[str] = re.compile(
    r"(?<=val PYDEVD_ASYNC_PLUGIN = \"\"\").*(?=\"\"\")",
    re.DOTALL,
)
PLUGIN_UPDATE_CODE_REGEX: Pattern[str] = re.compile(
    r"val PYDEVD_ASYNC_PLUGIN.*",
    re.DOTALL,
)


@click.group()
def entry_point():
    pass


@entry_point.command(name="plugin")
@click.option("--check", type=bool, is_flag=True, default=False)
def plugin_entry_point(check: bool) -> None:
    plugin: Path
    (plugin,) = Path.cwd().rglob("AsyncPyDebugUtils.kt")

    if check:
        code: str = PLUGIN_CHECK_CODE_REGEX.search(plugin.read_text("utf-8")).group()

        if code.strip() != generate():
            raise ValueError("Plugin code is outdated")
    else:
        async_pydevd_plugin = f'val PYDEVD_ASYNC_PLUGIN = """\n{generate()}\n""".trimStart()'

        plugin.write_text(
            PLUGIN_UPDATE_CODE_REGEX.sub("", plugin.read_text("utf-8")) + async_pydevd_plugin,
            "utf-8",
        )


README_SEPARATOR: str = "<!-- Plugin Examples -->"

README: Path = Path(__file__).parent.parent / "README.md"

IMAGES: list[str] = [
    "evaluate_expression.jpeg",
    "conditional_breakpoint.jpeg",
    "debugger_console.jpeg",
    "python_console.jpeg",
]


@entry_point.command(name="readme")
def readme_entry_point() -> None:
    readme: Path
    (readme,) = Path.cwd().glob("README.md")

    def generate_html_for_img(img: str) -> str:
        name, *_ = img.split(".")
        title = name.replace("_", " ").title()

        return textwrap.dedent(
            f"""
                <!-- Start - {title} -->
                <div>
                  <h2 align="center">{title}</h2>
                
                  <h1 align="center">
                    <img width="80%" alt="{name}" src="/images/{name}.jpeg">
                  </h1>
                </div>
                <!-- End - {title} -->
            """
        )

    main_content, *_ = readme.read_text("utf-8").split(README_SEPARATOR)

    readme.write_text(
        "\n".join(
            (
                main_content.rstrip(),
                "",
                README_SEPARATOR,
                "".join(generate_html_for_img(img) for img in IMAGES),
            )
        ),
        "utf-8",
    )


@entry_point.command(name="format")
def format_entry_point() -> None:
    subprocess.run(
        """\
        isort async-pydevd helpers
        black async-pydevd helpers -l 100
        """,
        shell=True,
    )


@entry_point.command(name="format-check")
@click.pass_context
def format_check_entry_point(ctx: click.Context) -> None:
    res = subprocess.run(
        """\
        isort async-pydevd helpers --check-only
        black async-pydevd helpers -l 100 --check
        """,
        shell=True,
    )
    ctx.exit(res.returncode)


__all__ = ["entry_point"]
