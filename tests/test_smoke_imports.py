"""Every module must load cleanly under a working env."""


def test_top_level():
    import flexposit
    assert flexposit.__version__
    for name in flexposit.__all__:
        assert hasattr(flexposit, name)


def test_all_modules_import():
    import flexposit.api
    import flexposit.core
    import flexposit.eval
    import flexposit.export
    import flexposit.formats
    import flexposit.models
    import flexposit.ppl
    import flexposit.utils
    import flexposit.quantizers.posit
    import flexposit.quantizers.int4
    import flexposit.quantizers.mxfp8
    import flexposit.mpq.channel_window
    import flexposit.mpq.layer
    import flexposit.sensitivity.ppl_probe
    import flexposit.sensitivity.ppl_probe_conv1d
    import flexposit.sensitivity.fisher


def test_version_matches_pyproject():
    import re
    from pathlib import Path

    import flexposit
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == flexposit.__version__
