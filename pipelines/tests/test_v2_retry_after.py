"""v2_to_v3_api_migration._retry_after_seconds: a 429's wait is always a usable sleep length."""
import pytest

m = pytest.importorskip("v2_to_v3_api_migration")


class R:
    def __init__(self, body):
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.mark.parametrize("body, expected", [
    ({"retry_after_seconds": 7}, 7),
    ({"retry_after_seconds": "12"}, 12),
    ({"retry_after_seconds": 2.6}, 2),
    ({"retry_after_seconds": -3}, 1),          # V2 sent this on 2026-10-27: sleep() raised ValueError
    ({"retry_after_seconds": 0}, 1),
    ({"retry_after_seconds": 99999}, 300),
    ({"retry_after_seconds": None}, 10),
    ({"retry_after_seconds": "soon"}, 10),
    ({}, 10),
    (ValueError("not json"), 10),
])
def test_retry_after_is_a_usable_sleep(body, expected):
    assert m._retry_after_seconds(R(body), 10) == expected
