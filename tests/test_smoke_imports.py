"""Every module must load cleanly under a working env."""


def test_top_level():
    import flexposit
    assert flexposit.__version__


def test_all_modules_import():
    import flexposit.ppl
    import flexposit.quantizers.posit
    import flexposit.quantizers.int4
    import flexposit.quantizers.mxfp8
    import flexposit.mpq.channel_window
    import flexposit.mpq.layer
    import flexposit.sensitivity.window
    import flexposit.sensitivity.conv1d
    import flexposit.sensitivity.fisher
