from typing import Any, AsyncContextManager, AsyncIterator, NoReturn

from async_pydevd.async_eval import async_eval
from pytest import mark, raises

pytestmark = mark.asyncio


class _MyException(Exception):
    pass


async def regular() -> int:
    return 10


async def generator() -> AsyncIterator[int]:
    for i in range(10):
        yield i


class _AsyncContextManager:
    async def __aenter__(self) -> int:
        return 10

    async def __aexit__(self, *_: Any):
        pass


def ctxmanager() -> AsyncContextManager[int]:
    return _AsyncContextManager()


def raise_exc() -> NoReturn:
    raise _MyException()


@mark.parametrize(
    "expr,result",
    [
        ("10", 10),
        ("regular", regular),
        ("await regular()", 10),
        ("[i async for i in generator()]", [*range(10)]),
        ("async with ctxmanager():\n    10", 10),
        ("await regular()\nawait regular() * 2", 20),
    ],
    ids=[
        "literal",
        "not-async",
        "await",
        "async-comprehension",
        "async-with",
        "multiline",
    ],
)
async def test_async_eval(expr, result):
    assert async_eval(expr) == result


@mark.parametrize(
    "expr,result",
    [
        ("a = 20", 20),
        ("a = regular", regular),
        ("a = await regular()", 10),
        ("a = [i async for i in generator()]", [*range(10)]),
        ("async with ctxmanager():\n    a = 10", 10),
    ],
    ids=[
        "literal",
        "not-async",
        "await",
        "async-comprehension",
        "async-with",
    ],
)
async def test_async_eval_modify_locals(expr, result):
    a = None
    async_eval(expr)
    assert a == result


async def test_eval_raise_exc():
    with raises(_MyException):
        async_eval("await raise_exc()")
