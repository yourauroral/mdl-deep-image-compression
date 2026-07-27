import mdlic


def test_public_package_imports_from_src_layout():
    assert mdlic.IGPT.__module__ == "mdlic.models.igpt"
    assert mdlic.CCIGPT.__module__ == "mdlic.models.cc_igpt"
    assert "/src/mdlic/" in mdlic.__file__.replace("\\", "/")
