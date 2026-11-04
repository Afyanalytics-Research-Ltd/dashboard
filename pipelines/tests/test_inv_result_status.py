"""V2 investigation result status → V3 (1 = published). Until 2026-10-27 every V2 result became
"preliminary", which V3 shows as pending — the patient history hid "View Results"."""
import pytest

m = pytest.importorskip("v2_to_v3_api_migration")


@pytest.mark.parametrize("v2, v3", [
    (1, "publish"), ("1", "publish"), (" 1 ", "publish"), (True, "publish"),
    (0, "preliminary"), ("0", "preliminary"), (None, "preliminary"), ("", "preliminary"), ("x", "preliminary"),
    ("publish", "publish"), ("Approved", "approved"), ("retracted", "retracted"), ("final", "final"),
])
def test_status_word(v2, v3):
    assert m._inv_result_status(v2) == v3


def test_coercion_uses_it():
    assert m._PER_KEY_COERCIONS["evaluation_inv_result"]["status"](1) == "publish"
