import ast
import inspect
import sys
import textwrap
import types
from itertools import takewhile
from typing import Any, Optional

from _pydevd_bundle.pydevd_save_locals import save_locals

_ASYNC_EVAL_CODE_TEMPLATE = """\
__locals__ = locals()

async def __async_exec_func():
    global __locals__
    locals().update(__locals__)
    try:
{}
    finally:
        __locals__.update(locals())

try:
    __async_exec_func_result__ = __import__('asyncio').get_event_loop().run_until_complete(__async_exec_func())
finally:
    del __locals__
    del __async_exec_func
"""


def _transform_to_async(expr: str) -> str:
    code = textwrap.indent(expr, " " * 8)
    code_without_return = _ASYNC_EVAL_CODE_TEMPLATE.format(code)

    node = ast.parse(code_without_return)
    last_node = node.body[1].body[2].body[-1]

    if isinstance(
        last_node,
        (
            ast.AsyncFor,
            ast.For,
            ast.Try,
            ast.If,
            ast.While,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        return code_without_return

    *others, last = code.splitlines(keepends=False)

    indent = sum(1 for _ in takewhile(str.isspace, last))
    last = " " * indent + f"return {last.lstrip()}"

    code_with_return = _ASYNC_EVAL_CODE_TEMPLATE.format("\n".join([*others, last]))

    try:
        compile(code_with_return, "<exec>", "exec")
        return code_with_return
    except SyntaxError:
        return code_without_return


# async equivalent of builtin eval function
def async_eval(expr: str, _globals: Optional[dict] = None, _locals: Optional[dict] = None) -> Any:
    caller: types.FrameType = inspect.currentframe().f_back

    if _locals is None:
        _locals = caller.f_locals

    if _globals is None:
        _globals = caller.f_globals

    code = _transform_to_async(expr)

    try:
        exec(code, _globals, _locals)
        return _locals.pop("__async_exec_func_result__")
    finally:
        save_locals(caller)


sys.__async_eval__ = async_eval
