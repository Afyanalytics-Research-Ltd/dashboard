# Modular per-domain ingestion

This package provides a shared ingestion lifecycle with domain-specific
extractors and validation. Domain configuration, dependency ordering, and
failure isolation are handled by the runner; domains can be added independently.

## Run

From the `pipelines` directory:

```shell
python -m ingestion_skeleton.ingestion.runner --config ingestion_skeleton/domains.yaml
```

Select one or more domains with `--only`. Their configured dependencies are
included automatically and run first. Disabled dependencies are reported as
skipped, and dependent domains are skipped rather than running without their
prerequisites.

## Add a domain

1. Add a configuration entry to `ingestion_skeleton/domains.yaml` with the
   source, primary keys, optional watermark column, and any dependencies.
2. If the generic SQL extractor is insufficient, add
   `ingestion_skeleton/ingestion/domains/<ingestor-key>.py`, subclass
   `BaseIngestor`, implement `extract()`, and register the class with
   `@register("<ingestor-key>")`. The runner loads modules by the configured
   ingestor key; no central module registry needs updating.
3. Override `validate()` for domain-specific checks or provide shared pipeline
   behavior through `Hooks`.

The base lifecycle writes raw and staged Parquet batches. Watermarks advance
only after reconciliation succeeds, so a failed run can safely be retried from
the previous watermark. `FileWatermarkStore` and `ParquetSink` are replaceable
implementations intended to be adapted to production storage.
