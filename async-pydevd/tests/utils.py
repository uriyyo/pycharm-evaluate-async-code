from typing import Any, AsyncContextManager, AsyncIterator, NoReturn


class MyException(Exception):
    pass


async def regular() -> int:
    return 10


async def generator() -> AsyncIterator[int]:
    for i in range(10):
        yield i


class AsyncContextManagerClass:
    async def __aenter__(self) -> int:
        return 10

    async def __aexit__(self, *_: Any):
        pass


def ctxmanager() -> AsyncContextManager[int]:
    return AsyncContextManagerClass()


def raise_exc() -> NoReturn:
    raise MyException()


__all__ = [
    "MyException",
    "regular",
    "generator",
    "AsyncContextManagerClass",
    "ctxmanager",
    "raise_exc",
]
