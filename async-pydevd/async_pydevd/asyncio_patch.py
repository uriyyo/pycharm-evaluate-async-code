import asyncio
import functools
import sys
from asyncio import AbstractEventLoop
from typing import Callable

try:
    from nest_asyncio import _patch_loop, apply
except ImportError:  # pragma: no cover
    pass


def _is_async_debug_available(loop=None) -> bool:
    if loop is None:
        loop = asyncio.get_event_loop()

    return loop.__class__.__module__.lstrip("_").startswith("asyncio")


def _patch_asyncio_set_get_new():
    if not _is_async_debug_available():
        return

    if sys.platform.lower().startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

    apply()

    def _patch_loop_if_not_patched(loop: AbstractEventLoop):
        if not hasattr(loop, "_nest_patched") and _is_async_debug_available(loop):
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
