from async_pydevd import FILES, generate


def test_generate():
    result = generate()

    assert '"""' not in result

    for f in FILES:
        normalized = (
            f.read_text("utf-8").replace('"""', "'''").replace("  # pragma: no cover", "").strip()
        )

        assert normalized in result

    assert not result.endswith(" ")
    assert not result.startswith(" ")
