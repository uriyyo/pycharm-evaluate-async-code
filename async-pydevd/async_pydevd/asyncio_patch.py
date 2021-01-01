import asyncio
import functools
import sys


def _patch_asyncio_set_get_new():
    if sys.platform.lower().startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    apply()

    def _patch_loop_if_not_patched(loop):
        if not hasattr(loop, "_nest_patched"):
            _patch_loop(loop)

    def _patch_asyncio_api(func):
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
    def set_loop_wrapper(loop):
        _patch_loop_if_not_patched(loop)
        _set_event_loop(loop)

    asyncio.set_event_loop = set_loop_wrapper


_patch_asyncio_set_get_new()
