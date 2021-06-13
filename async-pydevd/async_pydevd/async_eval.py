import ast
import inspect
import sys
import textwrap
import types
from itertools import takewhile
from typing import Any, Optional

from _pydevd_bundle.pydevd_save_locals import save_locals

_ASYNC_EVAL_CODE_TEMPLATE = textwrap.dedent(
    """\
__locals__ = locals()

async def __async_exec_func__():
    global __locals__
    locals().update(__locals__)
    try:
{}
    finally:
        __locals__.update(locals())

__ctx__ = None

try:
    async def __async_exec_func__(
        __async_exec_func__=__async_exec_func__,
        contextvars=__import__('contextvars'),
    ):
        try:
            return await __async_exec_func__()
        finally:
            global __ctx__
            __ctx__ = contextvars.copy_context()

except ImportError:
    pass

try:
    __async_exec_func_result__ = __import__('asyncio').get_event_loop().run_until_complete(__async_exec_func__())
finally:
    if __ctx__ is not None:
        for var in __ctx__:
            var.set(__ctx__[var])

        try:
            del var
        except NameError:
            pass

    del __ctx__
    del __locals__
    del __async_exec_func__

    try:
        del __builtins__
    except NameError:
        pass
"""
)


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
