import sys
from asyncio import AbstractEventLoop, BaseEventLoop, get_event_loop_policy
from platform import system

from pytest import mark


def _is_patched(loop: AbstractEventLoop) -> bool:
    return hasattr(loop, "_nest_patched")


class CustomEventLoop(BaseEventLoop):
    pass


def _test_asyncio_patch():
    from async_pydevd import asyncio_patch  # noqa # isort:skip
    from asyncio import get_event_loop, new_event_loop, set_event_loop  # isort:skip

    assert _is_patched(get_event_loop())
    assert _is_patched(new_event_loop())

    loop = CustomEventLoop()
    set_event_loop(loop)
    assert _is_patched(loop)


def test_asyncio_patch(run_in_process):
    run_in_process(_test_asyncio_patch)


def _test_windows_asyncio_policy():
    from async_pydevd import asyncio_patch  # noqa # isort:skip
    from asyncio.windows_events import WindowsSelectorEventLoopPolicy  # isort:skip

    assert isinstance(
        get_event_loop_policy(),
        WindowsSelectorEventLoopPolicy,
    )


@mark.skipif(
    not (system().lower() == "windows" and sys.version_info >= (3, 7)),
    reason="Only for Windows",
)
def test_windows_asyncio_policy(run_in_process):
    run_in_process(_test_windows_asyncio_policy)
