from pathlib import Path
from typing import Final

README_SEPARATOR: str = '<!-- Plugin Examples -->'

README: Final[Path] = Path(__file__).parent / 'README.md'

IMAGES: list[str] = [
    'evaluate_expression.jpeg',
    'conditional_breakpoint.jpeg',
    'debugger_console.jpeg',
    'python_console.jpeg',
]


def generate_html_for_img(img: str) -> str:
    name, *_ = img.split('.')
    title = name.replace('_', ' ').title()

    return f"""\

<!-- Start - {title} -->
<div>
  <h2 align="center">{title}</h2>

  <h1 align="center">
    <img width="80%" alt="{name}" src="/images/{name}.jpeg">
  </h1>
</div>
<!-- End - {title} -->
"""


main_content, *_ = README.read_text('utf-8').split(README_SEPARATOR)

README.write_text(
    "\n".join((
        main_content.rstrip(),
        "",
        README_SEPARATOR,
        "".join(generate_html_for_img(img) for img in IMAGES),
    )),
    encoding="utf-8",
)
