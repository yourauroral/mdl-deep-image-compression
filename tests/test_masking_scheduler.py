import pytest

from mdlic.masking_scheduler import MaskSchedule


def test_raster_chunks_are_balanced_deterministic_and_complete():
    first = MaskSchedule.raster_chunks(10, 3)
    second = MaskSchedule.raster_chunks(10, 3)

    assert first.groups == ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9))
    assert first == second
    assert first.sha256 == second.sha256
    assert sorted(index for group in first.groups for index in group) == list(range(10))
    assert max(map(len, first.groups)) - min(map(len, first.groups)) <= 1


def test_schedule_serialization_roundtrip_binds_exact_groups():
    schedule = MaskSchedule.raster_chunks(12, 4, token_layout="pixel-first-rgb")
    restored = MaskSchedule.from_dict(schedule.to_dict())
    reordered = MaskSchedule(
        name=schedule.name,
        num_tokens=schedule.num_tokens,
        groups=tuple(reversed(schedule.groups)),
        token_layout=schedule.token_layout,
    )

    assert restored == schedule
    assert restored.sha256 == schedule.sha256
    assert reordered.sha256 != schedule.sha256


def test_schedule_rejects_missing_duplicate_and_mismatched_serialized_groups():
    with pytest.raises(ValueError, match="exactly once"):
        MaskSchedule(name="bad", num_tokens=4, groups=((0, 1), (1, 2)))

    payload = MaskSchedule.raster_chunks(4, 2).to_dict()
    payload["num_groups"] = 3
    with pytest.raises(ValueError, match="num_groups"):
        MaskSchedule.from_dict(payload)

    payload = MaskSchedule.raster_chunks(4, 2).to_dict()
    payload["future_field"] = "must-not-be-ignored"
    with pytest.raises(ValueError, match="unknown"):
        MaskSchedule.from_dict(payload)


def test_teacher_forced_state_reveals_only_previous_groups():
    schedule = MaskSchedule.raster_chunks(6, 3)
    tokens = [10, 11, 12, 13, 14, 15]

    assert schedule.masked_state(tokens, 0, mask_token_id=256) == (256,) * 6
    assert schedule.masked_state(tokens, 1, mask_token_id=256) == (
        10, 11, 256, 256, 256, 256,
    )
    assert schedule.masked_state(tokens, 2, mask_token_id=256) == (
        10, 11, 12, 13, 256, 256,
    )


@pytest.mark.parametrize("num_tokens,num_groups", [(0, 1), (4, 0), (4, 5)])
def test_raster_chunks_reject_invalid_sizes(num_tokens, num_groups):
    with pytest.raises(ValueError):
        MaskSchedule.raster_chunks(num_tokens, num_groups)
