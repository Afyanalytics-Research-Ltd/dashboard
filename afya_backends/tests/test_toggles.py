import pytest

from afya_backends import toggles


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("AFYA_WAREHOUSE_BACKEND", "WAREHOUSE_BACKEND", "AFYA_LLM_PROVIDER", "LLM_PROVIDER",
              "LLM_PROVIDER_DEFAULT"):
        monkeypatch.delenv(k, raising=False)
    saved = list(toggles._sources)
    toggles._sources.clear()
    toggles.clear_cache()
    yield
    toggles._sources[:] = saved
    toggles.clear_cache()


def test_defaults():
    assert toggles.warehouse_backend() == "snowflake"
    assert toggles.llm_provider() == "api"


def test_registered_source_wins_over_deploy_default(monkeypatch):
    monkeypatch.setenv("WAREHOUSE_BACKEND", "snowflake")
    toggles.register_source(lambda s: "databend" if s == toggles.WAREHOUSE else None)
    assert toggles.warehouse_backend() == "databend"


def test_explicit_override_env_beats_sources(monkeypatch):
    toggles.register_source(lambda s: "databend")
    monkeypatch.setenv("AFYA_WAREHOUSE_BACKEND", "snowflake")
    assert toggles.warehouse_backend() == "snowflake"


def test_thread_override():
    with toggles.override(toggles.WAREHOUSE, "databend"):
        assert toggles.is_databend()
    assert not toggles.is_databend()


def test_broken_source_is_ignored():
    def boom(_):
        raise RuntimeError("db down")
    toggles.register_source(boom)
    assert toggles.warehouse_backend() == "snowflake"


@pytest.mark.parametrize("raw, want", [("Databend", "databend"), ("datablend", "databend"),
                                       ({"value": "databend"}, "databend"), ("nonsense", None),
                                       ('"snowflake"', "snowflake")])
def test_normalize(raw, want):
    assert toggles.normalize(toggles.WAREHOUSE, raw) == want


def test_legacy_llm_provider_env_is_not_the_platform_default(monkeypatch):
    # pipelines/.env has LLM_PROVIDER=ollama for the old CLI agent; that must
    # not silently flip the platform switch.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert toggles.llm_provider() == "api"
    monkeypatch.setenv("LLM_PROVIDER_DEFAULT", "ollama")
    toggles.clear_cache()
    assert toggles.llm_provider() == "ollama"


def test_llm_aliases():
    assert toggles.normalize(toggles.LLM, "local") == "ollama"
    assert toggles.normalize(toggles.LLM, "API keys") is None
    assert toggles.normalize(toggles.LLM, "api_keys") == "api"


def test_state_paths_only_change_on_databend(tmp_path):
    p = tmp_path / ".watermarks.json"
    assert toggles.state_path(p) == p
    with toggles.override(toggles.WAREHOUSE, "databend"):
        assert toggles.state_path(p).name == ".watermarks.databend.json"
        assert toggles.state_key("wm__x") == "wm__x__databend"
    assert toggles.state_key("wm__x") == "wm__x"
