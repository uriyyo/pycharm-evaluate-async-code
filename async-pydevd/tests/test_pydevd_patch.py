import sys
from unittest.mock import MagicMock

from pytest import fixture, mark

from .utils import ctxmanager  # noqa


def _as_async(code: str):
    return f"__import__('sys').__async_eval__({code!r}, globals(), locals())"


@fixture(autouse=True)
def _clear_modules():
    for name in [*sys.modules]:
        if name.startswith("pydevd") or name.startswith("async_pydevd"):
            del sys.modules[name]


params_mark = mark.parametrize(
    "code,result",
    [
        ("foo()",) * 2,
        (_as_async("await foo()"),) * 2,
        ("await foo()", _as_async("await foo()")),
    ],
)


@params_mark
def test_evaluate_expression(mocker, code, result):
    mock_eval: MagicMock = mocker.patch("_pydevd_bundle.pydevd_vars.evaluate_expression")
    mock_find_frame: MagicMock = mocker.patch("_pydevd_bundle.pydevd_vars.find_frame")

    from async_pydevd import pydevd_patch  # noqa # isort:skip
    from _pydevd_bundle.pydevd_vars import evaluate_expression  # isort:skip

    thread_id, frame_id = object(), object()
    evaluate_expression(thread_id, frame_id, code, True)

    do_exec = code != "await foo()"

    mock_eval.assert_called_once_with(thread_id, frame_id, result, do_exec)
    mock_find_frame.assert_called_once_with(thread_id, frame_id)


@params_mark
def test_line_breakpoint(code, result):
    from async_pydevd import pydevd_patch  # noqa # isort:skip
    from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint

    line = LineBreakpoint(line=0, func_name="test", condition=code, expression=code)

    assert line.condition == result
    assert line.expression == result


@params_mark
def test_console_integration(mocker, code, result):
    mock = mocker.patch("_pydevd_bundle.pydevd_console_integration.console_exec")

    from async_pydevd.pydevd_patch import console_exec

    thread_id, frame_id, dbg = object(), object(), object()

    console_exec(thread_id, frame_id, code, dbg)

    mock.assert_called_once_with(
        thread_id,
        frame_id,
        result,
        dbg,
    )


@params_mark
def test_command_run(mocker, code, result):
    from _pydev_bundle.pydev_console_types import CodeFragment, Command

    mock = mocker.MagicMock()

    command = Command(mock, CodeFragment(code))
    command.run()

    command.interpreter.runsource.assert_called_once_with(
        result,
        "<input>",
        "single",
    )


@params_mark
def test_make_code_async(code, result):
    from async_pydevd.pydevd_patch import make_code_async

    assert make_code_async(code) == result


@mark.parametrize(
    "code,result",
    [
        ("await foo()", True),
        ("[i async for i in range(10)]", True),
        ("foo()", False),
        ("__import__('sys').__async_eval__('await foo()')", False),
    ],
)
def test_is_code_async(code, result):
    from async_pydevd.pydevd_patch import is_async_code

    assert is_async_code(code) == result


# issue #6
def test_evaluate_expression_should_update_locals(mocker):
    def _with_locals():
        yield
        yield

    g = _with_locals()

    mocker.patch("_pydevd_bundle.pydevd_vars.find_frame", return_value=g.gi_frame)

    from async_pydevd.pydevd_patch import evaluate_expression

    evaluate_expression(
        object(),
        object(),
        """async with ctxmanager() as f:    pass""",
        True,
    )

    assert "f" in g.gi_frame.f_locals
    assert g.gi_frame.f_locals["f"] == 10
