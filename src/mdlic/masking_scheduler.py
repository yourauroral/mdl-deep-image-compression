"""Deterministic, serializable masking schedules for grouped codecs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEDULE_SCHEMA = "mdlic-mask-schedule-v1"
RASTER_CHUNKS_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class MaskSchedule:
    """An exact, data-independent partition and reveal order over token indices."""

    name: str
    num_tokens: int
    groups: tuple[tuple[int, ...], ...]
    token_layout: str = "pixel-first"
    algorithm_version: int = RASTER_CHUNKS_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("schedule name must be a non-empty string")
        _positive_int(self.num_tokens, "num_tokens")
        _positive_int(self.algorithm_version, "algorithm_version")
        if not isinstance(self.token_layout, str) or not self.token_layout:
            raise ValueError("token_layout must be a non-empty string")

        try:
            normalized = tuple(tuple(group) for group in self.groups)
        except TypeError as exc:
            raise ValueError("groups must be a sequence of index sequences") from exc
        if not normalized or any(not group for group in normalized):
            raise ValueError("groups must contain at least one non-empty group")

        flattened: list[int] = []
        for group_index, group in enumerate(normalized):
            for token_index in group:
                if not isinstance(token_index, int) or isinstance(token_index, bool):
                    raise ValueError(
                        f"groups[{group_index}] contains a non-integer index"
                    )
                if not 0 <= token_index < self.num_tokens:
                    raise ValueError(
                        f"token index {token_index} is outside [0,{self.num_tokens})"
                    )
                flattened.append(token_index)
        if len(flattened) != self.num_tokens or set(flattened) != set(range(self.num_tokens)):
            raise ValueError("groups must cover every token exactly once")
        object.__setattr__(self, "groups", normalized)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULE_SCHEMA,
            "name": self.name,
            "algorithm_version": self.algorithm_version,
            "token_layout": self.token_layout,
            "num_tokens": self.num_tokens,
            "num_groups": self.num_groups,
            "groups": [list(group) for group in self.groups],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaskSchedule":
        if not isinstance(value, Mapping) or value.get("schema") != SCHEDULE_SCHEMA:
            raise ValueError(f"schedule schema must be {SCHEDULE_SCHEMA!r}")
        expected_keys = {
            "schema",
            "name",
            "algorithm_version",
            "token_layout",
            "num_tokens",
            "num_groups",
            "groups",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unknown = sorted(set(value) - expected_keys)
            raise ValueError(
                f"schedule fields mismatch; missing={missing}, unknown={unknown}"
            )
        schedule = cls(
            name=value.get("name"),
            algorithm_version=value.get("algorithm_version"),
            token_layout=value.get("token_layout"),
            num_tokens=value.get("num_tokens"),
            groups=value.get("groups"),
        )
        if value.get("num_groups") != schedule.num_groups:
            raise ValueError("serialized num_groups does not match groups")
        return schedule

    @classmethod
    def raster_chunks(
        cls,
        num_tokens: int,
        num_groups: int,
        *,
        token_layout: str = "pixel-first",
    ) -> "MaskSchedule":
        num_tokens = _positive_int(num_tokens, "num_tokens")
        num_groups = _positive_int(num_groups, "num_groups")
        if num_groups > num_tokens:
            raise ValueError("num_groups cannot exceed num_tokens")
        groups = tuple(
            tuple(range(
                group_index * num_tokens // num_groups,
                (group_index + 1) * num_tokens // num_groups,
            ))
            for group_index in range(num_groups)
        )
        return cls(
            name="raster_chunks",
            num_tokens=num_tokens,
            groups=groups,
            token_layout=token_layout,
            algorithm_version=RASTER_CHUNKS_VERSION,
        )

    def revealed_before(self, group_index: int) -> tuple[int, ...]:
        self._validate_group_index(group_index)
        return tuple(
            token_index
            for group in self.groups[:group_index]
            for token_index in group
        )

    def masked_state(
        self,
        tokens: Sequence[int],
        group_index: int,
        *,
        mask_token_id: int,
    ) -> tuple[int, ...]:
        """Return the teacher-forced state at the start of one group."""
        self._validate_group_index(group_index)
        if len(tokens) != self.num_tokens:
            raise ValueError(
                f"token length {len(tokens)} does not match schedule {self.num_tokens}"
            )
        state = [mask_token_id] * self.num_tokens
        for token_index in self.revealed_before(group_index):
            state[token_index] = int(tokens[token_index])
        return tuple(state)

    def _validate_group_index(self, group_index: int) -> None:
        if (
            not isinstance(group_index, int)
            or isinstance(group_index, bool)
            or not 0 <= group_index < self.num_groups
        ):
            raise IndexError(
                f"group_index must be in [0,{self.num_groups}), got {group_index!r}"
            )
