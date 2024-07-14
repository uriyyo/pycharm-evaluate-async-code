package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.ParamsGroup
import com.intellij.openapi.application.PathManager
import com.intellij.openapi.projectRoots.Sdk
import com.jetbrains.python.psi.LanguageLevel
import java.io.File
import java.lang.reflect.Field
import java.lang.reflect.Method
import kotlin.io.path.Path
import kotlin.io.path.createTempFile

const val PYDEVD_ASYNC_DEBUG = "_pydevd_async_debug.py"
const val PLUGIN_NAME = "evaluate-async-code"

inline fun <reified R> loadClass(name: String): R? {
    return try {
        Class.forName(name)
            .getDeclaredConstructor()
            .also { it.isAccessible = true }
            .newInstance() as R
    } catch (e: ClassNotFoundException) {
        null
    }
}

interface MethodCall<R> {
    fun invoke(vararg args: Any?): R
}

fun allMethods(clazz: Class<*>): Sequence<Method> = sequence {
    var node = clazz
    while (node != Object::class.java) {
        node.declaredMethods.forEach { yield(it) }
        node = node.superclass
    }
}

inline fun <reified R> Any.getMethodByName(method: String): MethodCall<R> {
    val obj = this
    val methods = allMethods(javaClass)
        .filter { it.name == method }
        .onEach { it.isAccessible = true }
        .toList()

    return object : MethodCall<R> {
        override fun invoke(vararg args: Any?): R {
            for (m in methods) {
                try {
                    return m.invoke(obj, *args) as R
                } catch (_: IllegalArgumentException) {
                }
            }

            throw NoSuchMethodException(method)
        }
    }
}

fun Any.getField(name: String): Field = javaClass
    .declaredFields
    .filter { it.name == name }
    .onEach { it.isAccessible = true }
    .first()

inline fun <reified R> Any.getFieldVal(name: String): R = getField(name).get(this) as R

fun Any.setFieldValue(name: String, value: Any) = getField(name).set(this, value)

inline fun ignoreExc(body: () -> Unit) {
    try {
        body()
    } catch (_: Exception) {
    }
}

fun <T> (() -> T).memoize(): (() -> T) {
    var result: T? = null
    return {
        result = result ?: this()
        result!!
    }
}

val asyncPyDevScript: () -> File = {
    var script = Path(PathManager.getPluginsPath(), PLUGIN_NAME, PYDEVD_ASYNC_DEBUG).toFile()

    try {
        script.createNewFile()
    } catch (e: Exception) {
        script = createTempFile(suffix = ".py").toFile()
    }

    script.setReadable(true, false)
    script.setWritable(true, false)
    script.writeText(PYDEVD_ASYNC_PLUGIN)

    script
}.memoize()

val setupAsyncPyDevScript: () -> String = { "exec($PYDEVD_INLINE_ASYNC_PLUGIN, *([{}] * 2))" }.memoize()

fun ParamsGroup.addPyDevAsyncWork() {
    this.parametersList.addAt(0, asyncPyDevScript().absolutePath)
}

fun isSupportedVersion(version: String?): Boolean =
    version !== null && LanguageLevel
        .fromPythonVersion(version.split(" ").last())
        ?.isAtLeast(LanguageLevel.PYTHON37) == true

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

try:
    del sys, _patch_pydevd
except NameError:
    pass
""".trimIndent()
}.memoize()

val PYDEVD_INLINE_ASYNC_PLUGIN = """'import asyncio\nfrom asyncio import AbstractEventLoop\nfrom typing import Any\n\n\ndef is_trio_running() -> bool:\n    try:\n        from trio._core._run import GLOBAL_RUN_CONTEXT\n    except ImportError:  # pragma: no cover\n        return False\n\n    return hasattr(GLOBAL_RUN_CONTEXT, "runner")\n\n\ndef get_current_loop() -> AbstractEventLoop:  # pragma: no cover\n    try:\n        return asyncio.get_running_loop()\n    except RuntimeError:\n        return asyncio.new_event_loop()\n\n\ndef is_async_debug_available(loop: Any = None) -> bool:\n    if loop is None:\n        loop = get_current_loop()\n\n    return bool(loop.__class__.__module__.lstrip("_").startswith("asyncio"))\n\n\ndef verify_async_debug_available() -> None:\n    if not is_trio_running() and not is_async_debug_available():\n        cls = get_current_loop().__class__\n\n        raise RuntimeError(\n            f"Can not evaluate async code with event loop {cls.__module__}.{cls.__qualname__}. "\n            "Only native asyncio event loop can be used for async code evaluating.",\n        )\n\n\n__all__ = [\n    "get_current_loop",\n    "is_trio_running",\n    "is_async_debug_available",\n    "verify_async_debug_available",\n]\n\nimport ast\nimport inspect\nimport platform\nimport sys\nimport textwrap\nimport types\nfrom asyncio.tasks import _enter_task, _leave_task, current_task\nfrom concurrent.futures import ThreadPoolExecutor\nfrom contextvars import Context, copy_context\nfrom typing import (\n    Any,\n    Awaitable,\n    Callable,\n    Dict,\n    Optional,\n    Tuple,\n    TypeVar,\n    Union,\n    cast,\n    no_type_check,\n)\n\n_: Any\n\n\ndef is_pypy() -> bool:\n    return platform.python_implementation().lower() == "pypy"\n\n\ntry:\n    from _pydevd_bundle.pydevd_save_locals import save_locals as _save_locals\nexcept ImportError:  # pragma: no cover\n    import ctypes\n\n    try:\n        _ = ctypes.pythonapi\n\n        def _save_locals(frame: types.FrameType) -> None:\n            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))\n    except AttributeError:\n\n        def _save_locals(frame: types.FrameType) -> None:\n            pass\n\n\ndef save_locals(frame: types.FrameType) -> None:\n    if not is_pypy():\n        _save_locals(frame)\n\n\ndef _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover\n    return None\n\n\ntry:\n    _ = verify_async_debug_available  # type: ignore # noqa\nexcept NameError:  # pragma: no cover\n    try:\n        from async_eval.asyncio_patch import verify_async_debug_available\n    except ImportError:\n        verify_async_debug_available = _noop\n\ntry:\n    _ = get_current_loop  # type: ignore # noqa\nexcept NameError:  # pragma: no cover\n    try:\n        from async_eval.asyncio_patch import get_current_loop\n    except ImportError:\n        get_current_loop = _noop\n\ntry:\n    _ = is_trio_running  # type: ignore # noqa\nexcept NameError:  # pragma: no cover\n    try:\n        from async_eval.asyncio_patch import is_trio_running\n    except ImportError:\n        is_trio_running = _noop\n\n\n_ASYNC_EVAL_CODE_TEMPLATE = textwrap.dedent(\n    \'\'\'\\\nasync def __async_func__(_locals, _ctx=None):\n    async def __func_wrapper__(_locals):\n        locals().update(_locals)\n        try:\n            pass\n        finally:\n            _locals.update(locals())\n            _locals.pop("_ctx", None)\n            _locals.pop("_locals", None)\n\n    if _ctx:\n        for v in _ctx:\n            v.set(_ctx[v])\n\n    from contextvars import copy_context\n\n    try:\n      return False, await __func_wrapper__(_locals), copy_context()\n    except Exception as excpz:\n       return True, excpz, copy_context()\n\'\'\',\n)\n\n\ndef _compile_ast(node: ast.AST, filename: str = "<eval>", mode: str = "exec") -> types.CodeType:\n    return cast(types.CodeType, compile(node, filename, mode))  # type: ignore\n\n\nASTWithBody = Union[ast.Module, ast.With, ast.AsyncWith]\n\n\ndef _make_stmt_as_return(parent: ASTWithBody, root: ast.AST, filename: str) -> types.CodeType:\n    node = parent.body[-1]\n\n    if isinstance(node, ast.Expr):\n        parent.body[-1] = ast.copy_location(ast.Return(node.value), node)\n\n    try:\n        return _compile_ast(root, filename)\n    except (SyntaxError, TypeError):  # pragma: no cover  # TODO: found case to cover except body\n        parent.body[-1] = node\n        return _compile_ast(root, filename)\n\n\ndef _transform_to_async(code: str, filename: str) -> types.CodeType:\n    base = ast.parse(_ASYNC_EVAL_CODE_TEMPLATE)\n    module = ast.parse(code)\n\n    func: ast.AsyncFunctionDef = cast(ast.AsyncFunctionDef, cast(ast.AsyncFunctionDef, base.body[0]).body[0])\n    try_stmt: ast.Try = cast(ast.Try, func.body[-1])\n\n    try_stmt.body = module.body\n\n    parent: ASTWithBody = module\n    while isinstance(parent.body[-1], (ast.AsyncWith, ast.With)):\n        parent = cast(ASTWithBody, parent.body[-1])\n\n    return _make_stmt_as_return(parent, base, filename)\n\n\ndef _compile_async_func(\n    code: types.CodeType,\n    _locals: Dict[str, Any],\n    _globals: Dict[str, Any],\n) -> Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any, Context]]]:\n    exec(code, _globals, _locals)\n\n    return cast(\n        Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any, Context]]],\n        _locals.pop("__async_func__"),\n    )\n\n\nclass _AsyncNodeFound(Exception):\n    pass\n\n\nclass _AsyncCodeVisitor(ast.NodeVisitor):\n    @classmethod\n    def check(cls, code: str) -> bool:\n        try:\n            node = ast.parse(code)\n        except SyntaxError:\n            return False\n\n        try:\n            return bool(cls().visit(node))\n        except _AsyncNodeFound:\n            return True\n\n    def _ignore(self, _: ast.AST) -> Any:\n        return\n\n    def _done(self, _: Optional[ast.AST] = None) -> Any:\n        raise _AsyncNodeFound\n\n    def _visit_gen(self, node: Union[ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp]) -> Any:\n        if any(gen.is_async for gen in node.generators):\n            self._done()\n\n        super().generic_visit(node)\n\n    def _visit_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Any:\n        # check only args and decorators\n        for n in (node.args, *node.decorator_list):\n            super().generic_visit(n)\n\n    # special check for a function def\n    visit_AsyncFunctionDef = _visit_func\n    visit_FunctionDef = _visit_func\n\n    # no need to check class definitions\n    visit_ClassDef = _ignore  # type: ignore\n\n    # basic async statements\n    visit_AsyncFor = _done  # type: ignore\n    visit_AsyncWith = _done  # type: ignore\n    visit_Await = _done  # type: ignore\n\n    # all kind of a generator/comprehensions (they can be async)\n    visit_GeneratorExp = _visit_gen\n    visit_ListComp = _visit_gen\n    visit_SetComp = _visit_gen\n    visit_DictComp = _visit_gen\n\n\ndef is_async_code(code: str) -> bool:\n    return _AsyncCodeVisitor.check(code)\n\n\nT = TypeVar("T")\n\n\n@no_type_check\ndef _asyncio_run_coro(coro: Awaitable[T]) -> T:\n    loop = get_current_loop()\n\n    if not loop.is_running():\n        return loop.run_until_complete(coro)\n\n    current = current_task(loop)\n\n    t = loop.create_task(coro)\n\n    try:\n        if current is not None:\n            _leave_task(loop, current)\n\n        while not t.done():\n            loop._run_once()\n\n        return t.result()\n    finally:\n        if current is not None:\n            _enter_task(loop, current)\n\n\n@no_type_check\ndef _trio_run_coro(coro: Awaitable[T]) -> T:\n    import trio\n\n    async def _run() -> T:\n        return await coro\n\n    with ThreadPoolExecutor(max_workers=1) as pool:\n        return pool.submit(trio.run, _run).result()\n\n\n@no_type_check\ndef _run_coro(func: Callable[..., Awaitable[T]], _locals: Any) -> T:\n    if is_trio_running():\n        return _trio_run_coro(func(_locals, copy_context()))\n\n    return _asyncio_run_coro(func(_locals))\n\n\ndef _reflect_context(ctx: Context) -> None:\n    for v in ctx:\n        v.set(ctx[v])\n\n\n# async equivalent of builtin eval function\ndef async_eval(\n    code: str,\n    _globals: Optional[Dict[str, Any]] = None,\n    _locals: Optional[Dict[str, Any]] = None,\n    *,\n    filename: str = "<eval>",\n) -> Any:\n    verify_async_debug_available()\n\n    caller: types.FrameType = inspect.currentframe().f_back  # type: ignore\n\n    if _locals is None:\n        _locals = caller.f_locals\n\n    if _globals is None:\n        _globals = caller.f_globals\n\n    code_obj = _transform_to_async(code, filename)\n    func = _compile_async_func(code_obj, _locals, _globals)\n\n    try:\n        is_exc, result, ctx = _run_coro(func, _locals)\n\n        _reflect_context(ctx)\n\n        if is_exc:\n            raise result\n\n        return result\n    finally:\n        save_locals(caller)\n\n\nsys.__async_eval__ = async_eval  # type: ignore\n\n__all__ = ["async_eval", "is_async_code"]\n\nfrom typing import Any\n\n\ndef _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover\n    return False\n\n\ntry:  # pragma: no cover\n    # only for testing purposes\n    _ = is_async_code  # type: ignore  # noqa\n    _ = verify_async_debug_available  # type: ignore  # noqa\nexcept NameError:  # pragma: no cover\n    try:\n        from async_eval.async_eval import is_async_code\n        from async_eval.asyncio_patch import verify_async_debug_available\n    except ImportError:\n        is_async_code = _noop\n        verify_async_debug_available = _noop\n\n\ndef make_code_async(code: str) -> str:\n    if not code:\n        return code\n\n    original_code = code.replace("@" + "LINE" + "@", "\\n")\n\n    if is_async_code(original_code):\n        return f"__import__(\'sys\').__async_eval__({original_code!r}, globals(), locals())"\n\n    return code\n\n\n# 1. Add ability to evaluate async expression\nfrom _pydevd_bundle import pydevd_save_locals, pydevd_vars\n\noriginal_evaluate = pydevd_vars.evaluate_expression\n\n\ndef evaluate_expression(thread_id: object, frame_id: object, expression: str, doExec: bool) -> Any:\n    if is_async_code(expression):\n        verify_async_debug_available()\n        doExec = False\n\n    try:\n        return original_evaluate(thread_id, frame_id, make_code_async(expression), doExec)\n    finally:\n        frame = pydevd_vars.find_frame(thread_id, frame_id)\n\n        if frame is not None:\n            pydevd_save_locals.save_locals(frame)\n            del frame\n\n\npydevd_vars.evaluate_expression = evaluate_expression\n\n# 2. Add ability to use async breakpoint conditions\nfrom _pydevd_bundle.pydevd_breakpoints import LineBreakpoint\n\n\ndef normalize_line_breakpoint(line_breakpoint: LineBreakpoint) -> None:\n    line_breakpoint.expression = make_code_async(line_breakpoint.expression)\n    line_breakpoint.condition = make_code_async(line_breakpoint.condition)\n\n\noriginal_init = LineBreakpoint.__init__\n\n\ndef line_breakpoint_init(self: LineBreakpoint, *args: Any, **kwargs: Any) -> None:\n    original_init(self, *args, **kwargs)\n    normalize_line_breakpoint(self)\n\n\nLineBreakpoint.__init__ = line_breakpoint_init\n\n# Update old breakpoints\nimport gc\n\nfor obj in gc.get_objects():  # pragma: no cover\n    if isinstance(obj, LineBreakpoint):\n        normalize_line_breakpoint(obj)\n\n# 3. Add ability to use async code in console\nfrom _pydevd_bundle import pydevd_console_integration\n\noriginal_console_exec = pydevd_console_integration.console_exec\n\n\ndef console_exec(thread_id: object, frame_id: object, expression: str, dbg: Any) -> Any:\n    return original_console_exec(thread_id, frame_id, make_code_async(expression), dbg)\n\n\npydevd_console_integration.console_exec = console_exec\n\n# 4. Add ability to use async code\nfrom _pydev_bundle.pydev_console_types import Command\n\n\ndef command_run(self: Command) -> None:\n    text = make_code_async(self.code_fragment.text)\n    symbol = self.symbol_for_fragment(self.code_fragment)\n\n    self.more = self.interpreter.runsource(text, "<input>", symbol)\n\n\nCommand.run = command_run\n\nimport sys\nfrom runpy import run_path\n\nif __name__ == "__main__":  # pragma: no cover\n    run_path(sys.argv.pop(1), {}, "__main__")  # pragma: no cover\n'"""
val PYDEVD_ASYNC_PLUGIN = """
import asyncio
from asyncio import AbstractEventLoop
from typing import Any


def is_trio_running() -> bool:
    try:
        from trio._core._run import GLOBAL_RUN_CONTEXT
    except ImportError:  # pragma: no cover
        return False

    return hasattr(GLOBAL_RUN_CONTEXT, "runner")


def get_current_loop() -> AbstractEventLoop:  # pragma: no cover
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def is_async_debug_available(loop: Any = None) -> bool:
    if loop is None:
        loop = get_current_loop()

    return bool(loop.__class__.__module__.lstrip("_").startswith("asyncio"))


def verify_async_debug_available() -> None:
    if not is_trio_running() and not is_async_debug_available():
        cls = get_current_loop().__class__

        raise RuntimeError(
            f"Can not evaluate async code with event loop {cls.__module__}.{cls.__qualname__}. "
            "Only native asyncio event loop can be used for async code evaluating.",
        )


__all__ = [
    "get_current_loop",
    "is_trio_running",
    "is_async_debug_available",
    "verify_async_debug_available",
]

import ast
import inspect
import platform
import sys
import textwrap
import types
from asyncio.tasks import _enter_task, _leave_task, current_task
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

_: Any


def is_pypy() -> bool:
    return platform.python_implementation().lower() == "pypy"


try:
    from _pydevd_bundle.pydevd_save_locals import save_locals as _save_locals
except ImportError:  # pragma: no cover
    import ctypes

    try:
        _ = ctypes.pythonapi

        def _save_locals(frame: types.FrameType) -> None:
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
    except AttributeError:

        def _save_locals(frame: types.FrameType) -> None:
            pass


def save_locals(frame: types.FrameType) -> None:
    if not is_pypy():
        _save_locals(frame)


def _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover
    return None


try:
    _ = verify_async_debug_available  # type: ignore # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.asyncio_patch import verify_async_debug_available
    except ImportError:
        verify_async_debug_available = _noop

try:
    _ = get_current_loop  # type: ignore # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.asyncio_patch import get_current_loop
    except ImportError:
        get_current_loop = _noop

try:
    _ = is_trio_running  # type: ignore # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.asyncio_patch import is_trio_running
    except ImportError:
        is_trio_running = _noop


_ASYNC_EVAL_CODE_TEMPLATE = textwrap.dedent(
    '''\
async def __async_func__(_locals, _ctx=None):
    async def __func_wrapper__(_locals):
        locals().update(_locals)
        try:
            pass
        finally:
            _locals.update(locals())
            _locals.pop("_ctx", None)
            _locals.pop("_locals", None)

    if _ctx:
        for v in _ctx:
            v.set(_ctx[v])

    from contextvars import copy_context

    try:
      return False, await __func_wrapper__(_locals), copy_context()
    except Exception as excpz:
       return True, excpz, copy_context()
''',
)


def _compile_ast(node: ast.AST, filename: str = "<eval>", mode: str = "exec") -> types.CodeType:
    return cast(types.CodeType, compile(node, filename, mode))  # type: ignore


ASTWithBody = Union[ast.Module, ast.With, ast.AsyncWith]


def _make_stmt_as_return(parent: ASTWithBody, root: ast.AST, filename: str) -> types.CodeType:
    node = parent.body[-1]

    if isinstance(node, ast.Expr):
        parent.body[-1] = ast.copy_location(ast.Return(node.value), node)

    try:
        return _compile_ast(root, filename)
    except (SyntaxError, TypeError):  # pragma: no cover  # TODO: found case to cover except body
        parent.body[-1] = node
        return _compile_ast(root, filename)


def _transform_to_async(code: str, filename: str) -> types.CodeType:
    base = ast.parse(_ASYNC_EVAL_CODE_TEMPLATE)
    module = ast.parse(code)

    func: ast.AsyncFunctionDef = cast(ast.AsyncFunctionDef, cast(ast.AsyncFunctionDef, base.body[0]).body[0])
    try_stmt: ast.Try = cast(ast.Try, func.body[-1])

    try_stmt.body = module.body

    parent: ASTWithBody = module
    while isinstance(parent.body[-1], (ast.AsyncWith, ast.With)):
        parent = cast(ASTWithBody, parent.body[-1])

    return _make_stmt_as_return(parent, base, filename)


def _compile_async_func(
    code: types.CodeType,
    _locals: Dict[str, Any],
    _globals: Dict[str, Any],
) -> Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any, Context]]]:
    exec(code, _globals, _locals)

    return cast(
        Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any, Context]]],
        _locals.pop("__async_func__"),
    )


class _AsyncNodeFound(Exception):
    pass


class _AsyncCodeVisitor(ast.NodeVisitor):
    @classmethod
    def check(cls, code: str) -> bool:
        try:
            node = ast.parse(code)
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


T = TypeVar("T")


@no_type_check
def _asyncio_run_coro(coro: Awaitable[T]) -> T:
    loop = get_current_loop()

    if not loop.is_running():
        return loop.run_until_complete(coro)

    current = current_task(loop)

    t = loop.create_task(coro)

    try:
        if current is not None:
            _leave_task(loop, current)

        while not t.done():
            loop._run_once()

        return t.result()
    finally:
        if current is not None:
            _enter_task(loop, current)


@no_type_check
def _trio_run_coro(coro: Awaitable[T]) -> T:
    import trio

    async def _run() -> T:
        return await coro

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(trio.run, _run).result()


@no_type_check
def _run_coro(func: Callable[..., Awaitable[T]], _locals: Any) -> T:
    if is_trio_running():
        return _trio_run_coro(func(_locals, copy_context()))

    return _asyncio_run_coro(func(_locals))


def _reflect_context(ctx: Context) -> None:
    for v in ctx:
        v.set(ctx[v])


# async equivalent of builtin eval function
def async_eval(
    code: str,
    _globals: Optional[Dict[str, Any]] = None,
    _locals: Optional[Dict[str, Any]] = None,
    *,
    filename: str = "<eval>",
) -> Any:
    verify_async_debug_available()

    caller: types.FrameType = inspect.currentframe().f_back  # type: ignore

    if _locals is None:
        _locals = caller.f_locals

    if _globals is None:
        _globals = caller.f_globals

    code_obj = _transform_to_async(code, filename)
    func = _compile_async_func(code_obj, _locals, _globals)

    try:
        is_exc, result, ctx = _run_coro(func, _locals)

        _reflect_context(ctx)

        if is_exc:
            raise result

        return result
    finally:
        save_locals(caller)


sys.__async_eval__ = async_eval  # type: ignore

__all__ = ["async_eval", "is_async_code"]

from typing import Any


def _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover
    return False


try:  # pragma: no cover
    # only for testing purposes
    _ = is_async_code  # type: ignore  # noqa
    _ = verify_async_debug_available  # type: ignore  # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.async_eval import is_async_code
        from async_eval.asyncio_patch import verify_async_debug_available
    except ImportError:
        is_async_code = _noop
        verify_async_debug_available = _noop


def make_code_async(code: str) -> str:
    if not code:
        return code

    original_code = code.replace("@" + "LINE" + "@", "\n")

    if is_async_code(original_code):
        return f"__import__('sys').__async_eval__({original_code!r}, globals(), locals())"

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