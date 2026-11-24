import pytest

from afya_backends import toggles


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("AFYA_WAREHOUSE_BACKEND", "WAREHOUSE_BACKEND", "AFYA_LLM_PROVIDER", "LLM_PROVIDER",
              "LLM_PROVIDER_DEFAULT", "AFYA_TOGGLE_URL", "AFYA_INTERNAL_API_TOKEN"):
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


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.body


def _no_airflow(monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, "airflow", raising=False)
    toggles._http_last.clear()


def test_streamlit_asks_django_and_keeps_the_last_answer_when_django_is_down(monkeypatch):
    import requests
    _no_airflow(monkeypatch)
    monkeypatch.setenv("WAREHOUSE_BACKEND", "snowflake")
    monkeypatch.setenv("AFYA_TOGGLE_URL", "http://web:8000/core/api/warehouse/translate/")
    monkeypatch.setenv("AFYA_INTERNAL_API_TOKEN", "t0k")
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append((url, headers))
        return _Resp({"warehouse_backend": "databend", "llm_provider": "api"})
    monkeypatch.setattr(requests, "get", get)
    assert toggles.warehouse_backend() == "databend"
    assert calls == [("http://web:8000/core/api/warehouse/translate/", {"X-Afya-Token": "t0k"})]

    def down(*a, **k):
        raise requests.ConnectionError("web restarting")
    monkeypatch.setattr(requests, "get", down)
    toggles.clear_cache()
    assert toggles.warehouse_backend() == "databend"          # not the deploy default


def test_http_source_is_not_used_inside_django_or_without_a_token(monkeypatch):
    import requests
    _no_airflow(monkeypatch)
    monkeypatch.setenv("AFYA_TOGGLE_URL", "http://web:8000/x/")
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP expected")))
    assert toggles.warehouse_backend() == "snowflake"         # no token: no call
    monkeypatch.setenv("AFYA_INTERNAL_API_TOKEN", "t0k")
    toggles.register_source(lambda s: None)                   # Django registered (setting unset)
    toggles.clear_cache()
    assert toggles.warehouse_backend() == "snowflake"


def test_http_source_defaults_on_only_inside_streamlit(monkeypatch):
    import sys
    monkeypatch.delenv("AFYA_TOGGLE_URL", raising=False)
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    assert toggles._toggle_url() is None
    monkeypatch.setitem(sys.modules, "streamlit", object())
    assert toggles._toggle_url() == toggles.DEFAULT_TOGGLE_URL
    monkeypatch.setenv("AFYA_TOGGLE_URL", "off")
    assert toggles._toggle_url() is None
