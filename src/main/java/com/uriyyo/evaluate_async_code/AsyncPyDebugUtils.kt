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

val PYDEVD_ASYNC_PLUGIN = """
import asyncio
import asyncio.events as events
import os
import sys
import threading
from heapq import heappop


def apply(loop=None):
    '''Patch asyncio to make its event loop reentrent.'''
    loop = loop or asyncio.get_event_loop()
    if not isinstance(loop, asyncio.BaseEventLoop):
        raise ValueError('Can\'t patch loop of type %s' % type(loop))
    if getattr(loop, '_nest_patched', None):
        # already patched
        return
    _patch_asyncio()
    _patch_loop(loop)
    _patch_task()
    _patch_handle()
    _patch_tornado()


def _patch_asyncio():
    '''
    Patch asyncio module to use pure Python tasks and futures,
    use module level _current_tasks, all_tasks and patch run method.
    '''
    def run(future, *, debug=False):
        loop = asyncio.get_event_loop()
        loop.set_debug(debug)
        return loop.run_until_complete(future)

    if sys.version_info >= (3, 6, 0):
        asyncio.Task = asyncio.tasks._CTask = asyncio.tasks.Task = \
            asyncio.tasks._PyTask
        asyncio.Future = asyncio.futures._CFuture = asyncio.futures.Future = \
            asyncio.futures._PyFuture
    if sys.version_info < (3, 7, 0):
        asyncio.tasks._current_tasks = asyncio.tasks.Task._current_tasks  # noqa
        asyncio.all_tasks = asyncio.tasks.Task.all_tasks  # noqa
    if not hasattr(asyncio, '_run_orig'):
        asyncio._run_orig = getattr(asyncio, 'run', None)
        asyncio.run = run


def _patch_loop(loop):
    '''Patch loop to make it reentrent.'''

    def run_forever(self):
        if sys.version_info >= (3, 7, 0):
            set_coro_tracking = self._set_coroutine_origin_tracking
        else:
            set_coro_tracking = self._set_coroutine_wrapper

        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = events._get_running_loop()
        set_coro_tracking(self._debug)
        self._thread_id = threading.get_ident()

        if self._asyncgens is not None:
            old_agen_hooks = sys.get_asyncgen_hooks()
            sys.set_asyncgen_hooks(
                firstiter=self._asyncgen_firstiter_hook,
                finalizer=self._asyncgen_finalizer_hook)
        try:
            events._set_running_loop(self)
            while True:
                self._run_once()
                if self._stopping:
                    break
        finally:
            self._stopping = False
            self._thread_id = old_thread_id
            events._set_running_loop(old_running_loop)
            set_coro_tracking(False)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(*old_agen_hooks)

    def run_until_complete(self, future):
        old_thread_id = self._thread_id
        old_running_loop = events._get_running_loop()
        try:
            self._check_closed()
            self._thread_id = threading.get_ident()
            events._set_running_loop(self)
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
        finally:
            self._thread_id = old_thread_id
            events._set_running_loop(old_running_loop)

    def _run_once(self):
        '''
        Simplified re-implementation of asyncio's _run_once that
        runs handles as they become ready.
        '''
        now = self.time()
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)

        timeout = (
            0 if ready or self._stopping
            else min(max(0, scheduled[0]._when - now), 86400) if scheduled
            else 0.01 if self._is_proactorloop
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

    def _check_running(self):
        '''Do not throw exception if loop is already running.'''
        pass

    cls = loop.__class__
    cls._run_once_orig = cls._run_once
    cls._run_once = _run_once
    cls._run_forever_orig = cls.run_forever
    cls.run_forever = run_forever
    cls._run_until_complete_orig = cls.run_until_complete
    cls.run_until_complete = run_until_complete
    cls._check_running = _check_running
    cls._check_runnung = _check_running  # typo in Python 3.7 source
    cls._nest_patched = True
    cls._is_proactorloop = (
            os.name == 'nt' and issubclass(cls, asyncio.ProactorEventLoop))


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


def _patch_handle():
    '''Patch Handle to allow recursive calls.'''

    def update_from_context(ctx):
        '''Copy context ctx to currently active context.'''
        for var in ctx:
            var.set(ctx[var])

    def run(self):
        '''
        Run the callback in a sub-context, then copy any sub-context vars
        over to the Handle's context.
        '''
        try:
            ctx = self._context.copy()
            ctx.run(self._callback, *self._args)
            if ctx:
                self._context.run(update_from_context, ctx)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            cb = format_helpers._format_callback_source(
                self._callback, self._args)
            msg = 'Exception in callback {}'.format(cb)
            context = {
                'message': msg,
                'exception': exc,
                'handle': self,
            }
            if self._source_traceback:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)
        self = None

    if sys.version_info >= (3, 7, 0):
        from asyncio import format_helpers
        events.Handle._run = run


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
from typing import Callable

try:
    from nest_asyncio import _patch_loop, apply
except ImportError:
    pass


def _patch_asyncio_set_get_new():
    if sys.platform.lower().startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

    apply()

    def _patch_loop_if_not_patched(loop: AbstractEventLoop):
        if not hasattr(loop, "_nest_patched"):
            _patch_loop(loop)

    def _patch_asyncio_api(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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

    asyncio.set_event_loop = set_loop_wrapper


_patch_asyncio_set_get_new()

import ast
import inspect
import sys
import textwrap
import types
from itertools import takewhile
from typing import Any, Optional

from _pydevd_bundle.pydevd_save_locals import save_locals

_ASYNC_EVAL_CODE_TEMPLATE = '''\
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
'''


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

from typing import Any


def is_async_code(code: str) -> bool:
    # TODO: use node visitor to check if code contains async/await
    return "__async_eval__" not in code and ("await" in code or "async" in code)


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


def line_breakpoint_init(self: LineBreakpoint, *args, **kwargs):
    original_init(self, *args, **kwargs)
    normalize_line_breakpoint(self)


LineBreakpoint.__init__ = line_breakpoint_init

# 3. Add ability to use async code in console
from _pydevd_bundle import pydevd_console_integration

original_console_exec = pydevd_console_integration.console_exec


def console_exec(thread_id: object, frame_id: object, expression: str, dbg) -> Any:
    return original_console_exec(thread_id, frame_id, make_code_async(expression), dbg)


pydevd_console_integration.console_exec = console_exec

# 4. Add ability to use async code
from _pydev_bundle.pydev_console_types import Command


def command_run(self) -> None:
    text = make_code_async(self.code_fragment.text)
    symbol = self.symbol_for_fragment(self.code_fragment)

    self.more = self.interpreter.runsource(text, "<input>", symbol)


Command.run = command_run

import sys
from runpy import run_path

run_path(sys.argv.pop(1), {}, "__main__")
""".trimStart()