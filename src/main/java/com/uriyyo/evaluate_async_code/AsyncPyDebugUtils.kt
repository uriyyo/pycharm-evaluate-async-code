package com.uriyyo.evaluate_async_code

import com.intellij.execution.configurations.ParamsGroup
import com.intellij.openapi.application.PathManager
import com.intellij.openapi.projectRoots.Sdk
import com.jetbrains.python.psi.LanguageLevel
import java.io.File
import kotlin.io.path.Path
import kotlin.io.path.createTempFile

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

try:
    del sys, _patch_pydevd
except NameError:
    pass
""".trimIndent()
}.memoize()

val PYDEVD_ASYNC_PLUGIN = """
import asyncio
import asyncio.events as events
import os
import sys
import threading
from contextlib import contextmanager, suppress
from heapq import heappop


def apply(loop=None):
    '''Patch asyncio to make its event loop reentrant.'''
    _patch_asyncio()
    _patch_task()
    _patch_tornado()

    loop = loop or asyncio.get_event_loop()
    _patch_loop(loop)


def _patch_asyncio():
    '''
    Patch asyncio module to use pure Python tasks and futures,
    use module level _current_tasks, all_tasks and patch run method.
    '''
    def run(main, *, debug=False):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _patch_loop(loop)
        loop.set_debug(debug)
        task = asyncio.ensure_future(main)
        try:
            return loop.run_until_complete(task)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    loop.run_until_complete(task)

    def _get_event_loop(stacklevel=3):
        loop = events._get_running_loop()
        if loop is None:
            loop = events.get_event_loop_policy().get_event_loop()
        return loop

    if hasattr(asyncio, '_nest_patched'):
        return
    if sys.version_info >= (3, 6, 0):
        asyncio.Task = asyncio.tasks._CTask = asyncio.tasks.Task = \
            asyncio.tasks._PyTask
        asyncio.Future = asyncio.futures._CFuture = asyncio.futures.Future = \
            asyncio.futures._PyFuture
    if sys.version_info < (3, 7, 0):
        asyncio.tasks._current_tasks = asyncio.tasks.Task._current_tasks
        asyncio.all_tasks = asyncio.tasks.Task.all_tasks
    if sys.version_info >= (3, 9, 0):
        events._get_event_loop = events.get_event_loop = \
            asyncio.get_event_loop = _get_event_loop
        _get_event_loop
    asyncio.run = run
    asyncio._nest_patched = True


def _patch_loop(loop):
    '''Patch loop to make it reentrant.'''

    def run_forever(self):
        with manage_run(self), manage_asyncgens(self):
            while True:
                self._run_once()
                if self._stopping:
                    break
        self._stopping = False

    def run_until_complete(self, future):
        with manage_run(self):
            f = asyncio.ensure_future(future, loop=self)
            if f is not future:
                f._log_destroy_pending = False
            while not f.done():
                self._run_once()
                if self._stopping:
                    break
            if not f.done():
                raise RuntimeError(
                    'Event loop stopped before Future completed.')
            return f.result()

    def _run_once(self):
        '''
        Simplified re-implementation of asyncio's _run_once that
        runs handles as they become ready.
        '''
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)

        timeout = (
            0 if ready or self._stopping
            else min(max(
                scheduled[0]._when - self.time(), 0), 86400) if scheduled
            else None)
        event_list = self._selector.select(timeout)
        self._process_events(event_list)

        end_time = self.time() + self._clock_resolution
        while scheduled and scheduled[0]._when < end_time:
            handle = heappop(scheduled)
            ready.append(handle)

        for _ in range(len(ready)):
            if not ready:
                break
            handle = ready.popleft()
            if not handle._cancelled:
                handle._run()
        handle = None

    @contextmanager
    def manage_run(self):
        '''Set up the loop for running.'''
        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = events._get_running_loop()
        try:
            self._thread_id = threading.get_ident()
            events._set_running_loop(self)
            self._num_runs_pending += 1
            if self._is_proactorloop:
                if self._self_reading_future is None:
                    self.call_soon(self._loop_self_reading)
            yield
        finally:
            self._thread_id = old_thread_id
            events._set_running_loop(old_running_loop)
            self._num_runs_pending -= 1
            if self._is_proactorloop:
                if (self._num_runs_pending == 0
                        and self._self_reading_future is not None):
                    ov = self._self_reading_future._ov
                    self._self_reading_future.cancel()
                    if ov is not None:
                        self._proactor._unregister(ov)
                    self._self_reading_future = None

    @contextmanager
    def manage_asyncgens(self):
        if not hasattr(sys, 'get_asyncgen_hooks'):
            # Python version is too old.
            return
        old_agen_hooks = sys.get_asyncgen_hooks()
        try:
            self._set_coroutine_origin_tracking(self._debug)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(
                    firstiter=self._asyncgen_firstiter_hook,
                    finalizer=self._asyncgen_finalizer_hook)
            yield
        finally:
            self._set_coroutine_origin_tracking(False)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(*old_agen_hooks)

    def _check_running(self):
        '''Do not throw exception if loop is already running.'''
        pass

    if hasattr(loop, '_nest_patched'):
        return
    if not isinstance(loop, asyncio.BaseEventLoop):
        raise ValueError('Can\'t patch loop of type %s' % type(loop))
    cls = loop.__class__
    cls.run_forever = run_forever
    cls.run_until_complete = run_until_complete
    cls._run_once = _run_once
    cls._check_running = _check_running
    cls._check_runnung = _check_running  # typo in Python 3.7 source
    cls._num_runs_pending = 0
    cls._is_proactorloop = (
            os.name == 'nt' and issubclass(cls, asyncio.ProactorEventLoop))
    if sys.version_info < (3, 7, 0):
        cls._set_coroutine_origin_tracking = cls._set_coroutine_wrapper
    cls._nest_patched = True


def _patch_task():
    '''Patch the Task's step and enter/leave methods to make it reentrant.'''

    def step(task, exc=None):
        curr_task = curr_tasks.get(task._loop)
        try:
            step_orig(task, exc)
        finally:
            if curr_task is None:
                curr_tasks.pop(task._loop, None)
            else:
                curr_tasks[task._loop] = curr_task

    Task = asyncio.Task
    if hasattr(Task, '_nest_patched'):
        return
    if sys.version_info >= (3, 7, 0):

        def enter_task(loop, task):
            curr_tasks[loop] = task

        def leave_task(loop, task):
            curr_tasks.pop(loop, None)

        asyncio.tasks._enter_task = enter_task
        asyncio.tasks._leave_task = leave_task
        curr_tasks = asyncio.tasks._current_tasks
        step_orig = Task._Task__step
        Task._Task__step = step
    else:
        curr_tasks = Task._current_tasks
        step_orig = Task._step
        Task._step = step
    Task._nest_patched = True


def _patch_tornado():
        '''
        If tornado is imported before nest_asyncio, make tornado aware of
        the pure-Python asyncio Future.
        '''
        if 'tornado' in sys.modules:
            import tornado.concurrent as tc
            tc.Future = asyncio.Future
            if asyncio.Future not in tc.FUTURES:
                tc.FUTURES += (asyncio.Future,)

import asyncio
import functools
import sys
from asyncio import AbstractEventLoop
from typing import Any, Callable, Optional

try:  # pragma: no cover
    _ = _patch_loop  # noqa
    _ = apply  # noqa
except NameError:
    try:
        from nest_asyncio import _patch_loop, apply
    except ImportError:  # pragma: no cover

        def _noop(*_: Any, **__: Any) -> None:
            pass

        _patch_loop = apply = _noop


try:
    from trio._core._run import GLOBAL_RUN_CONTEXT
except ImportError:  # pragma: no cover
    GLOBAL_RUN_CONTEXT = object()


def is_trio_not_running() -> bool:
    return not hasattr(GLOBAL_RUN_CONTEXT, "runner")


def get_current_loop() -> Optional[Any]:  # pragma: no cover
    try:
        return asyncio.get_running_loop()  # type: ignore
    except RuntimeError:
        return asyncio.new_event_loop()
    except AttributeError:  # Only for 3.6 case, can be removed in future
        return asyncio.get_event_loop()


def is_async_debug_available(loop: Any = None) -> bool:
    if loop is None:
        loop = get_current_loop()

    return bool(loop.__class__.__module__.lstrip("_").startswith("asyncio"))


def verify_async_debug_available() -> None:
    if not is_trio_not_running():
        raise RuntimeError(
            "Can not evaluate async code with trio event loop. "
            "Only native asyncio event loop can be used for async code evaluating."
        )

    if not is_async_debug_available():
        cls = get_current_loop().__class__

        raise RuntimeError(
            f"Can not evaluate async code with event loop {cls.__module__}.{cls.__qualname__}. "
            "Only native asyncio event loop can be used for async code evaluating."
        )


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

__all__ = [
    "patch_asyncio",
    "get_current_loop",
    "is_async_debug_available",
    "verify_async_debug_available",
]

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


def _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover
    return None


try:
    _ = verify_async_debug_available  # noqa
except NameError:  # pragma: no cover
    try:
        from async_eval.asyncio_patch import verify_async_debug_available
    except ImportError:
        verify_async_debug_available = _noop

try:
    _ = apply  # noqa
except NameError:  # pragma: no cover
    try:
        from nest_asyncio import apply
    except ImportError:
        apply = _noop


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
    base: ast.Module = ast.parse(_ASYNC_EVAL_CODE_TEMPLATE)
    module: ast.Module = cast(ast.Module, _parse_code(code))

    func: ast.AsyncFunctionDef = cast(ast.AsyncFunctionDef, base.body[1])
    try_stmt: ast.Try = cast(ast.Try, func.body[-1])

    try_stmt.body = module.body

    parent: ASTWithBody = module
    while isinstance(parent.body[-1], (ast.AsyncWith, ast.With)):
        parent = cast(ASTWithBody, parent.body[-1])

    return _make_stmt_as_return(parent, base, filename)


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
def async_eval(
    code: str,
    _globals: Optional[dict] = None,
    _locals: Optional[dict] = None,
    *,
    filename: str = "<eval>",
) -> Any:
    verify_async_debug_available()
    apply()  # double check that loop is patched

    caller: types.FrameType = inspect.currentframe().f_back  # type: ignore

    if _locals is None:
        _locals = caller.f_locals

    if _globals is None:
        _globals = caller.f_globals

    code_obj = _transform_to_async(code, filename)

    try:
        exec(code_obj, _globals, _locals)
        return _locals.pop("__async_exec_func_result__")
    finally:
        save_locals(caller)


sys.__async_eval__ = async_eval  # type: ignore

__all__ = ["async_eval", "is_async_code"]

from typing import Any


def _noop(*_: Any, **__: Any) -> Any:  # pragma: no cover
    return False


try:  # pragma: no cover
    # only for testing purposes
    _ = is_async_code  # noqa
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