import pytest

from mdlic.completion import row_aligned_keep_tokens


@pytest.mark.parametrize(
    ("keep_frac", "expected_rows"),
    [
        (0.001, 1),
        (0.5, 16),
        (0.51, 16),
        (0.53, 17),
        (1.0, 32),
    ],
)
def test_keep_boundary_is_a_complete_rgb_raster_row(keep_frac, expected_rows):
    keep = row_aligned_keep_tokens(
        keep_frac,
        image_size=32,
        channels=3,
        total_tokens=32 * 32 * 3,
    )

    assert keep == expected_rows * 32 * 3
    assert keep % (32 * 3) == 0


@pytest.mark.parametrize("keep_frac", [0.0, -0.1, 1.1, float("nan")])
def test_keep_boundary_rejects_invalid_fraction(keep_frac):
    with pytest.raises(ValueError, match="keep_frac"):
        row_aligned_keep_tokens(
            keep_frac,
            image_size=32,
            channels=3,
            total_tokens=32 * 32 * 3,
        )


def test_keep_boundary_rejects_inconsistent_token_geometry():
    with pytest.raises(ValueError, match="does not describe"):
        row_aligned_keep_tokens(
            0.5,
            image_size=32,
            channels=3,
            total_tokens=3071,
        )
