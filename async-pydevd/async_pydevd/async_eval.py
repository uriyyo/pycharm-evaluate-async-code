import ast
import sys
import textwrap


# Async equivalent of builtin eval function
def async_eval(expr: str, _globals: dict = None, _locals: dict = None):
    if _locals is None:
        _locals = {}

    if _globals is None:
        _globals = {}

    expr = textwrap.indent(expr, "    ")
    expr = f"async def _():\n{expr}"

    parsed_stmts = ast.parse(expr).body[0]
    for node in parsed_stmts.body:
        ast.increment_lineno(node)

    last_stmt = parsed_stmts.body[-1]

    if isinstance(last_stmt, ast.Expr):
        return_expr = ast.copy_location(ast.Return(last_stmt), last_stmt)
        return_expr.value = return_expr.value.value
        parsed_stmts.body[-1] = return_expr

    parsed_fn = ast.parse(
        f"""\
async def __async_exec_func__(__locals__=__locals__):
    try:
        pass
    finally:
        __locals__.update(locals())
        del __locals__['__locals__']

import asyncio

__async_exec_func_result__ = asyncio.get_event_loop().run_until_complete(__async_exec_func__())
    """
    )

    parsed_fn.body[0].body[0].body = parsed_stmts.body

    try:
        code = compile(parsed_fn, filename="<ast>", mode="exec")
    except (SyntaxError, TypeError):
        parsed_stmts.body[-1] = last_stmt
        parsed_fn.body[0].body[0].body = parsed_stmts.body
        code = compile(parsed_fn, filename="<ast>", mode="exec")

    _updated_locals = {
        **_locals,
        "__locals__": _locals,
    }
    _updated_globals = {
        **_globals,
        **_updated_locals,
    }

    exec(code, _updated_globals, _updated_locals)
    return _updated_locals["__async_exec_func_result__"]


sys.__async_eval__ = async_eval
