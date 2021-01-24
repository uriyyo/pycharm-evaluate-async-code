from asyncio import AbstractEventLoop
from multiprocessing import Process
from typing import Callable

import nest_asyncio
from pytest import fixture

# issue #7
nest_asyncio._patch_handle = lambda: None


@fixture
def event_loop(event_loop) -> AbstractEventLoop:
    nest_asyncio.apply(event_loop)
    return event_loop


@fixture
def run_in_process():
    def _run_in_process(func: Callable, timeout: int = 20) -> None:
        p = Process(target=func)
        p.start()
        p.join(timeout)

        assert not p.exitcode

    return _run_in_process
