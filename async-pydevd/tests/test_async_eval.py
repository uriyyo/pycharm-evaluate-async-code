import textwrap

from async_pydevd.async_eval import async_eval
from pytest import fixture, mark, raises

from .utils import MyException, ctxmanager, generator, raise_exc, regular  # noqa  # isort:skip

try:
    import contextvars
except ImportError:
    contextvars = None


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


async def test_async_eval_dont_leak_internal_vars():
    _globals = _locals = {}
    async_eval("10", _globals, _locals)

    assert not _globals
    assert not _locals


if contextvars:
    ctx_var = contextvars.ContextVar("ctx_var")


@mark.skipif(
    contextvars is None,
    reason="contextvars is not available",
)
class TestContextVars:
    @fixture(autouse=True)
    def reset_var(self):
        ctx_var.set(0)

    def test_ctx_get(self):
        assert async_eval("ctx_var.get()") == 0

    def test_ctx_set(self):
        async_eval("ctx_var.set(10)")
        assert ctx_var.get() == 10

    # issue #7
    def test_ctx_var_reset(self):
        # fmt: off
        async_eval(textwrap.dedent("""
        from asyncio import sleep
        token = ctx_var.set(10)
        await sleep(0)  # switch to different task
        ctx_var.reset(token)
        """))
        # fmt: on

        assert ctx_var.get() == 0
