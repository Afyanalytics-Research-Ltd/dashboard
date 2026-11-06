"""Contract tests every per-domain ingestor implementation should satisfy."""
import pandas as pd
import pytest

from ingestion_skeleton.ingestion.base import BaseIngestor
from ingestion_skeleton.ingestion.config import DomainConfig
from ingestion_skeleton.ingestion.context import IngestResult
from ingestion_skeleton.ingestion.hooks import Hooks
from ingestion_skeleton.ingestion.registry import load_ingestor_cls, register
from ingestion_skeleton.ingestion.runner import run_all
from ingestion_skeleton.ingestion.stores import FileWatermarkStore, ParquetSink


class FakeIngestor(BaseIngestor):
    def extract(self, ctx):
        df = pd.DataFrame({"id": [1, 2, 3], "updated_at": ["2026-10-01", "2026-10-02", "2026-10-03"]})
        if ctx.watermark_in:
            df = df[df["updated_at"] > ctx.watermark_in]
        return df


def make(tmp_path, cls=FakeIngestor, name="fake"):
    cfg = DomainConfig(name=name, source_dsn_env="X", source_table="t", source_schema_ref="a",
                       target_schema_ref="b", primary_keys=["id"], watermark_column="updated_at")
    return cls(cfg, watermarks=FileWatermarkStore(tmp_path / "wm.json"), sink=ParquetSink(tmp_path / "data"))


def test_success_and_incremental(tmp_path):
    ing = make(tmp_path)
    r1 = ing.run()
    assert r1.status == "success" and r1.counts["loaded"] == 3
    assert r1.watermark_out == "2026-10-03"
    r2 = ing.run()                      # nothing new after the watermark
    assert r2.status == "success" and r2.counts["extracted"] == 0


def test_failure_is_contained_and_watermark_not_advanced(tmp_path):
    class Boom(FakeIngestor):
        def extract(self, ctx):
            raise RuntimeError("source down")
    ing = make(tmp_path, Boom)
    r = ing.run()
    assert r.status == "failed" and "source down" in r.error
    assert ing.watermarks.get("fake") is None


def test_validation_catches_missing_key_column(tmp_path):
    class NoKey(FakeIngestor):
        def extract(self, ctx):
            return pd.DataFrame({"updated_at": ["2026-10-01"]})
    r = make(tmp_path, NoKey).run()
    assert r.status == "failed" and "missing required columns" in r.error


def test_reconcile_failure_does_not_advance_watermark(tmp_path):
    def fail_reconcile(ctx):
        raise RuntimeError("reconciliation failed")

    ing = make(tmp_path)
    ing.hooks = Hooks(reconcile=fail_reconcile)

    result = ing.run()

    assert result.status == "failed"
    assert ing.watermarks.get("fake") is None


def test_selecting_domain_includes_dependencies_and_runs_in_order(tmp_path):
    order = []

    class FakeRunnerIngestor:
        def __init__(self, cfg, hooks=None):
            self.cfg = cfg

        def run(self):
            order.append(self.cfg.name)
            return IngestResult(self.cfg.name, "run", "success")

    register("test_root")(FakeRunnerIngestor)
    register("test_parent")(FakeRunnerIngestor)
    register("test_child")(FakeRunnerIngestor)
    config = tmp_path / "domains.yaml"
    config.write_text(
        """domains:
  root:
    source_dsn_env: SRC
    source_table: root
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    ingestor: test_root
  parent:
    source_dsn_env: SRC
    source_table: parent
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    ingestor: test_parent
    depends_on: [root]
  child:
    source_dsn_env: SRC
    source_table: child
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    ingestor: test_child
    depends_on: [parent]
  independent:
    source_dsn_env: SRC
    source_table: independent
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    ingestor: test_independent
""",
        encoding="utf-8",
    )

    results = run_all(str(config), only=["child"])

    assert order == ["root", "parent", "child"]
    assert [result.domain for result in results] == ["root", "parent", "child"]


def test_disabled_dependency_blocks_selected_domain(tmp_path):
    config = tmp_path / "domains.yaml"
    config.write_text(
        """domains:
  parent:
    source_dsn_env: SRC
    source_table: parent
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    enabled: false
  child:
    source_dsn_env: SRC
    source_table: child
    source_schema_ref: source
    target_schema_ref: target
    primary_keys: [id]
    depends_on: [parent]
""",
        encoding="utf-8",
    )

    results = run_all(str(config), only=["child"])

    assert [(result.domain, result.status) for result in results] == [
        ("parent", "skipped"),
        ("child", "skipped"),
    ]
    assert "parent" in results[1].error


def test_unknown_selected_domain_is_rejected(tmp_path):
    config = tmp_path / "domains.yaml"
    config.write_text("domains: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown domain"):
        run_all(str(config), only=["missing"])


def test_new_ingestor_module_is_loaded_by_key(tmp_path, monkeypatch):
    from ingestion_skeleton.ingestion import domains

    module_file = tmp_path / "custom.py"
    module_file.write_text(
        """from ingestion_skeleton.ingestion.registry import register

@register("custom")
class CustomIngestor:
    pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(domains, "__path__", [str(tmp_path)])

    cls = load_ingestor_cls("custom")

    assert cls.__name__ == "CustomIngestor"


def test_duplicate_ingestor_registration_is_rejected():
    register("duplicate_test")(type("OriginalIngestor", (), {}))

    with pytest.raises(ValueError, match="already registered"):
        register("duplicate_test")(type("ReplacementIngestor", (), {}))
