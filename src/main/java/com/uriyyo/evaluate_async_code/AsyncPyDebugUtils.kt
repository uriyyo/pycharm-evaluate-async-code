package com.uriyyo.evaluate_async_code

import com.jetbrains.python.psi.LanguageLevel

val String.isAsyncCode
    get() = "async" in this || "await" in this

fun isSupportedVersion(version: String?): Boolean =
        version !== null && LanguageLevel
                .fromPythonVersion(version.split(" ").last())
                ?.isAtLeast(LanguageLevel.PYTHON36) == true


val PLUGIN = """
def __patcher__():
    import gc
    import ast
    import asyncio
    import asyncio.events as events
    import sys
    from heapq import heappop

    if getattr(sys, '__async_patched__', False):
        return

    # Thanks to https://github.com/erdewit/nest_asyncio/blob/master/nest_asyncio.py
    def patch_asyncio():
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

            def run_until_complete(self, future):
                self._check_closed()
                events._set_running_loop(self)
                f = asyncio.ensure_future(future, loop=self)
                if f is not future:
                    f._log_destroy_pending = False
                while not f.done():
                    self._run_once()
                    if self._stopping:
                        break
                if not f.done():
                    raise RuntimeError('Event loop stopped before Future completed.')
                return f.result()

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

                timeout = 0 if ready or self._stopping \
                    else min(max(0, scheduled[0]._when - now), 10) if scheduled \
                    else None
                event_list = self._selector.select(timeout)
                self._process_events(event_list)

                while scheduled and scheduled[0]._when < now + self._clock_resolution:
                    handle = heappop(scheduled)
                    ready.append(handle)

                while ready:
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
            cls._run_until_complete_orig = cls.run_until_complete
            cls.run_until_complete = run_until_complete
            cls._check_running = _check_running
            cls._check_runnung = _check_running  # typo in Python 3.7 source
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
            if sys.version_info >= (3, 7, 0):

                def enter_task(loop, task):
                    curr_tasks[loop] = task

                def leave_task(loop, task):
                    del curr_tasks[loop]

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
                except Exception as exc:
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

        apply()

    patch_asyncio()

    def make_code_async(code: str) -> str:
        if not code:
            return code

        if "__async_eval__" not in code and ("await" in code or "async" in code):
            code = code.replace("@" + "LINE" + "@", "\n")
            return f"__import('sys')__.__async_eval__({code!r}, globals(), locals())"

        return code

    # Async equivalent of builtin eval function
    def async_eval(expr: str, _globals: dict = None, _locals: dict = None):
        if _locals is None:
            _locals = {}

        if _globals is None:
            _globals = {}

        parsed_stmts = ast.parse(expr)
        for node in parsed_stmts.body:
            ast.increment_lineno(node)

        last_stmt = parsed_stmts.body[-1]

        if isinstance(last_stmt, ast.Expr):
            return_expr = ast.copy_location(ast.Return(last_stmt), last_stmt)
            return_expr.value = return_expr.value.value
            parsed_stmts.body[-1] = return_expr

        parsed_fn = ast.parse(
f'''
async def __async_exec_func__(__locals__=__locals__):
    try:
        pass
    finally:
        __locals__.update(locals())
        del __locals__['__locals__']

import asyncio

__async_exec_func_result__ = asyncio.get_event_loop().run_until_complete(__async_exec_func__())
''')

        parsed_fn.body[0].body[0].body = parsed_stmts.body

        try:
            code = compile(parsed_fn, filename="<ast>", mode="exec")
        except (SyntaxError, TypeError):
            parsed_stmts.body[-1] = last_stmt
            parsed_fn.body[0].body[0].body = parsed_stmts.body
            code = compile(parsed_fn, filename="<ast>", mode="exec")

        _updated_locals = {
            **_locals,
            '__locals__': _locals,
        }

        exec(code, _globals, _updated_locals)
        return _updated_locals['__async_exec_func_result__']

    sys.__async_eval__ = async_eval

    def _pydevd_patch():
        # Add ability to evaluate async expression
        from _pydevd_bundle import pydevd_vars

        original_evaluate = pydevd_vars.evaluate_expression

        def evaluate_expression(thread_id: int, frame_id: int, expression: str, doExec: bool):
            return original_evaluate(thread_id, frame_id, expression, doExec)

        pydevd_vars.evaluate_expression = evaluate_expression

        # Add ability to use async breakpoint conditions
        from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint

        def normalize_line_breakpoint(line_breakpoint: LineBreakpoint) -> None:
            line_breakpoint.expression = make_code_async(line_breakpoint.expression)
            line_breakpoint.condition = make_code_async(line_breakpoint.condition)

        original_init = LineBreakpoint.__init__

        def line_breakpoint_init(self: LineBreakpoint, *args, **kwargs):
            original_init(self, *args, **kwargs)
            normalize_line_breakpoint(self)

        LineBreakpoint.__init__ = line_breakpoint_init

        for obj in gc.get_objects():
            if isinstance(obj, LineBreakpoint):
                normalize_line_breakpoint(obj)

        # Add ability to use async code in console
        from _pydevd_bundle import pydevd_console_integration

        original_console_exec = pydevd_console_integration.console_exec

        def console_exec(thread_id: int, frame_id: int, expression: str, dbg):
            return original_console_exec(thread_id, frame_id, make_code_async(expression), dbg)

        pydevd_console_integration.console_exec = console_exec

    _pydevd_patch()
    sys.__async_patched__ = True

__patcher__()
del __patcher__
"""