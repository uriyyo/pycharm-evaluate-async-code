from async_pydevd.async_eval import async_eval
from pytest import mark, raises

from .utils import MyException, ctxmanager, generator, raise_exc, regular  # noqa

pytestmark = mark.asyncio


@mark.parametrize(
    "expr,result",
    [
        ("10", 10),
        ("regular", regular),
        ("await regular()", 10),
        ("[i async for i in generator()]", [*range(10)]),
        ("async with ctxmanager():\n    10", 10),
        ("await regular()\nawait regular() * 2", 20),
        ("async for i in generator():\n    i * 2", None),
    ],
    ids=[
        "literal",
        "not-async",
        "await",
        "async-comprehension",
        "async-with",
        "multiline",
        "async-for",
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
        ("async for i in generator():\n    a = i", 9),
    ],
    ids=[
        "literal",
        "not-async",
        "await",
        "async-comprehension",
        "async-with",
        "async-for",
    ],
)
async def test_async_eval_modify_locals(expr, result):
    a = None
    async_eval(expr)
    assert a == result


async def test_eval_raise_exc():
    with raises(MyException):
        async_eval("await raise_exc()")
