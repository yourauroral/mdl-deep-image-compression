from scripts.pixelsnail_paramcount import pixelcnnpp_params, pixelsnail_params


def test_pixelsnail_parameter_estimate_matches_documented_configuration():
    total, parts = pixelsnail_params(residual_blocks=4)

    assert total == 91_371_816
    assert sum(parts.values()) == total
    assert pixelcnnpp_params() == 55_349_320
