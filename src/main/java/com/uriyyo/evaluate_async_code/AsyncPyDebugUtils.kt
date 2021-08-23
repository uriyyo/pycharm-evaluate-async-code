package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.ParamsGroup
import com.intellij.openapi.application.PathManager
import com.intellij.openapi.projectRoots.Sdk
import com.jetbrains.python.psi.LanguageLevel
import java.io.File
import java.nio.file.Paths

const val PYDEVD_ASYNC_DEBUG = "_pydevd_async_debug.py"
const val PLUGIN_NAME = "evaluate-async-code"

fun <T> (() -> T).memoize(): (() -> T) {
    var result: T? = null
    return {
        result = result ?: this()
        result!!
    }
}

val asyncPyDevScript: () -> File = {
    var script = Paths.get(PathManager.getPluginsPath(), PLUGIN_NAME, PYDEVD_ASYNC_DEBUG).toFile()

    try {
        script.createNewFile()
    } catch (e: Exception) {
        script = createTempFile(suffix = ".py")
    }

    script.setReadable(true, false)
    script.setWritable(true, false)
    script.writeText(PYDEVD_ASYNC_PLUGIN)

    script
}.memoize()


fun ParamsGroup.addPyDevAsyncWork() {
    this.parametersList.addAt(0, asyncPyDevScript().absolutePath)
}

fun isSupportedVersion(version: String?): Boolean =
        version !== null && LanguageLevel
                .fromPythonVersion(version.split(" ").last())
                ?.isAtLeast(LanguageLevel.PYTHON36) == true

fun Sdk.whenSupport(block: () -> Unit) {
    if (isSupportedVersion(this.versionString))
        block()
}

val pydevd_async_init: () -> String = {
    """
def _patch_pydevd(__name__=None):
${PYDEVD_ASYNC_PLUGIN.prependIndent("    ")}

import sys

if not hasattr(sys, "__async_eval__"):
    _patch_pydevd()
""".trimIndent()
}.memoize()

val PYDEVD_ASYNC_PLUGIN = """
import asyncio
import functools
import sys
from asyncio import AbstractEventLoop
from typing import Any, Callable

try:
    from nest_asyncio import _patch_loop, apply
except ImportError:  # pragma: no cover

    def _noop(*args: Any, **kwargs: Any) -> None:
        pass

    _patch_loop = apply = _noop


def is_async_debug_available(loop: Any = None) -> bool:
    if loop is None:
        loop = asyncio.get_event_loop()

    return bool(loop.__class__.__module__.lstrip("_").startswith("asyncio"))


def patch_asyncio() -> None:
    if hasattr(sys, "__async_eval_patched__"):  # pragma: no cover
        return

    if not is_async_debug_available():  # pragma: no cover
        return

    if sys.platform.lower().startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except AttributeError:
            pass

    apply()

    def _patch_loop_if_not_patched(loop: AbstractEventLoop) -> None:
        if not hasattr(loop, "_nest_patched") and is_async_debug_available(loop):
            _patch_loop(loop)

    def _patch_asyncio_api(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = func(*args, **kwargs)
            _patch_loop_if_not_patched(loop)
            return loop

        return wrapper

    asyncio.get_event_loop = _patch_asyncio_api(asyncio.get_event_loop)
    asyncio.new_event_loop = _patch_asyncio_api(asyncio.new_event_loop)

    _set_event_loop = asyncio.set_event_loop

    @functools.wraps(asyncio.set_event_loop)
    def set_loop_wrapper(loop: AbstractEventLoop) -> None:
        _patch_loop_if_not_patched(loop)
        _set_event_loop(loop)

    asyncio.set_event_loop = set_loop_wrapper  # type: ignore
    sys.__async_eval_patched__ = True  # type: ignore


patch_asyncio()

__all__ = ["patch_asyncio", "is_async_debug_available"]

import ast
import inspect
import sys
import textwrap
import types
from typing import Any, Iterable, Optional, Union, cast

try:
    from _pydevd_bundle.pydevd_save_locals import save_locals
except ImportError:  # pragma: no cover

    import ctypes

    def save_locals(frame: types.FrameType) -> None:
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))


_ASYNC_EVAL_CODE_TEMPLATE = textwrap.dedent(
    '''\
__locals__ = locals()

async def __async_exec_func__():
    global __locals__
    locals().update(__locals__)
    try:
        pass
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
'''
)

if sys.version_info < (3, 7):

    def _parse_code(code: str) -> ast.AST:
        code = f"async def _():\n{textwrap.indent(code, '    ')}"
        func, *_ = cast(Iterable[ast.AsyncFunctionDef], ast.parse(code).body)
        return ast.Module(func.body)


else:
    _parse_code = ast.parse


def _compile_ast(node: ast.AST, filename: str = "<eval>", mode: str = "exec") -> types.CodeType:
    return cast(types.CodeType, compile(node, filename, mode))


ASTWithBody = Union[ast.Module, ast.With, ast.AsyncWith]


def _make_stmt_as_return(parent: ASTWithBody, root: ast.AST) -> types.CodeType:
    node = parent.body[-1]

    if isinstance(node, ast.Expr):
        parent.body[-1] = ast.copy_location(ast.Return(node.value), node)

    try:
        return _compile_ast(root)
    except (SyntaxError, TypeError):  # pragma: no cover  # TODO: found case to cover except body
        parent.body[-1] = node
        return _compile_ast(root)


def _transform_to_async(code: str) -> types.CodeType:
    base: ast.Module = ast.parse(_ASYNC_EVAL_CODE_TEMPLATE)
    module: ast.Module = cast(ast.Module, _parse_code(code))

    func: ast.AsyncFunctionDef = cast(ast.AsyncFunctionDef, base.body[1])
    try_stmt: ast.Try = cast(ast.Try, func.body[-1])

    try_stmt.body = module.body

    parent: ASTWithBody = module
    while isinstance(parent.body[-1], (ast.AsyncWith, ast.With)):
        parent = cast(ASTWithBody, parent.body[-1])

    return _make_stmt_as_return(parent, base)


class _AsyncNodeFound(Exception):
    pass


class _AsyncCodeVisitor(ast.NodeVisitor):
    @classmethod
    def check(cls, code: str) -> bool:
        try:
            node = _parse_code(code)
        except SyntaxError:
            return False

        try:
            return bool(cls().visit(node))
        except _AsyncNodeFound:
            return True

    def _ignore(self, _: ast.AST) -> Any:
        return

    def _done(self, _: Optional[ast.AST] = None) -> Any:
        raise _AsyncNodeFound

    def _visit_gen(self, node: Union[ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp]) -> Any:
        if any(gen.is_async for gen in node.generators):
            self._done()

        super().generic_visit(node)

    def _visit_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Any:
        # check only args and decorators
        for n in (node.args, *node.decorator_list):
            super().generic_visit(n)

    # special check for a function def
    visit_AsyncFunctionDef = _visit_func
    visit_FunctionDef = _visit_func

    # no need to check class definitions
    visit_ClassDef = _ignore  # type: ignore

    # basic async statements
    visit_AsyncFor = _done  # type: ignore
    visit_AsyncWith = _done  # type: ignore
    visit_Await = _done  # type: ignore

    # all kind of a generator/comprehensions (they can be async)
    visit_GeneratorExp = _visit_gen
    visit_ListComp = _visit_gen
    visit_SetComp = _visit_gen
    visit_DictComp = _visit_gen


def is_async_code(code: str) -> bool:
    return _AsyncCodeVisitor.check(code)


# async equivalent of builtin eval function
def async_eval(code: str, _globals: Optional[dict] = None, _locals: Optional[dict] = None) -> Any:
    caller: types.FrameType = inspect.currentframe().f_back  # type: ignore

    if _locals is None:
        _locals = caller.f_locals

    if _globals is None:
        _globals = caller.f_globals

    code_obj = _transform_to_async(code)

    try:
        exec(code_obj, _globals, _locals)
        return _locals.pop("__async_exec_func_result__")
    finally:
        save_locals(caller)


sys.__async_eval__ = async_eval  # type: ignore

__all__ = ["async_eval", "is_async_code"]

import asyncio
from typing import Any


def _noop(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    return True


try:  # pragma: no cover
    # only for testing purposes
    _ = is_async_debug_available  # noqa
    _ = is_async_code  # type: ignore  # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.async_eval import is_async_code
        from async_eval.asyncio_patch import is_async_debug_available
    except ImportError:

        is_async_code = _noop
        is_async_debug_available = _noop

try:
    from trio._core._run import GLOBAL_RUN_CONTEXT

    def is_trio_running() -> bool:
        return hasattr(GLOBAL_RUN_CONTEXT, "runner")


except ImportError:  # pragma: no cover
    is_trio_running = _noop


def verify_async_debug_available() -> None:
    if is_trio_running():
        raise RuntimeError(
            "Can not evaluate async code with trio event loop. "
            "Only native asyncio event loop can be used for async code evaluating."
        )

    if not is_async_debug_available():
        cls = asyncio.get_event_loop().__class__

        raise RuntimeError(
            f"Can not evaluate async code with event loop {cls.__module__}.{cls.__qualname__}. "
            "Only native asyncio event loop can be used for async code evaluating."
        )


def make_code_async(code: str) -> str:
    if code and is_async_code(code):
        code = code.replace("@" + "LINE" + "@", "\n")
        return f"__import__('sys').__async_eval__({code!r}, globals(), locals())"

    return code


# 1. Add ability to evaluate async expression
from _pydevd_bundle import pydevd_save_locals, pydevd_vars

original_evaluate = pydevd_vars.evaluate_expression


def evaluate_expression(thread_id: object, frame_id: object, expression: str, doExec: bool) -> Any:
    if is_async_code(expression):
        verify_async_debug_available()
        doExec = False

    try:
        return original_evaluate(thread_id, frame_id, make_code_async(expression), doExec)
    finally:
        frame = pydevd_vars.find_frame(thread_id, frame_id)

        if frame is not None:
            pydevd_save_locals.save_locals(frame)
            del frame


pydevd_vars.evaluate_expression = evaluate_expression

# 2. Add ability to use async breakpoint conditions
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint


def normalize_line_breakpoint(line_breakpoint: LineBreakpoint) -> None:
    line_breakpoint.expression = make_code_async(line_breakpoint.expression)
    line_breakpoint.condition = make_code_async(line_breakpoint.condition)


original_init = LineBreakpoint.__init__


def line_breakpoint_init(self: LineBreakpoint, *args: Any, **kwargs: Any) -> None:
    original_init(self, *args, **kwargs)
    normalize_line_breakpoint(self)


LineBreakpoint.__init__ = line_breakpoint_init

# Update old breakpoints
import gc

for obj in gc.get_objects():  # pragma: no cover
    if isinstance(obj, LineBreakpoint):
        normalize_line_breakpoint(obj)

# 3. Add ability to use async code in console
from _pydevd_bundle import pydevd_console_integration

original_console_exec = pydevd_console_integration.console_exec


def console_exec(thread_id: object, frame_id: object, expression: str, dbg: Any) -> Any:
    return original_console_exec(thread_id, frame_id, make_code_async(expression), dbg)


pydevd_console_integration.console_exec = console_exec

# 4. Add ability to use async code
from _pydev_bundle.pydev_console_types import Command


def command_run(self: Command) -> None:
    text = make_code_async(self.code_fragment.text)
    symbol = self.symbol_for_fragment(self.code_fragment)

    self.more = self.interpreter.runsource(text, "<input>", symbol)


Command.run = command_run

import sys
from runpy import run_path

if __name__ == "__main__":  # pragma: no cover
    run_path(sys.argv.pop(1), {}, "__main__")  # pragma: no cover

""".trimStart()