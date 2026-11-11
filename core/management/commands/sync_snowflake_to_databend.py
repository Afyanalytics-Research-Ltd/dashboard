"""Copy Snowflake schemas (structure + data) into Databend.

    python manage.py sync_snowflake_to_databend --schemas STAGING,REPORTING
    python manage.py sync_snowflake_to_databend --all
    python manage.py sync_snowflake_to_databend --schemas RAW --tables EVENTS_RAW --replace

Run it when Snowflake is reachable (e.g. after the account is restored) to
bring Snowflake-built schemas such as HOSPITALS.STAGING / REPORTING and
pre-switch history into Databend.  Resumable (see afya_backends/warehouse/sync.py).
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Copy Snowflake schemas (tables, views, data) into Databend."

    def add_arguments(self, parser):
        parser.add_argument("--schemas", default="", help="Comma-separated Snowflake schemas, e.g. STAGING,REPORTING")
        parser.add_argument("--all", action="store_true", help="Every schema in the database")
        parser.add_argument("--tables", default="", help="Only these tables/views (comma-separated)")
        parser.add_argument("--database", default="", help="Snowflake database (default SNOWFLAKE_DATABASE / HOSPITALS)")
        parser.add_argument("--replace", action="store_true", help="Reload tables a previous run already copied")
        parser.add_argument("--batch-rows", type=int, default=200_000)
        parser.add_argument("--state-file", default=".snowflake_to_databend_sync.json")
        parser.add_argument("--admin", action="store_true", help="Use SNOWFLAKE_ADMIN_* credentials")

    def handle(self, *args, **o):
        import snowflake.connector
        from afya_backends.warehouse import sql_translate as tr
        from afya_backends.warehouse.dbapi import Connection
        from afya_backends.warehouse.sync import Syncer
        from core.warehouse_creds import snowflake_kwargs

        if not o["schemas"] and not o["all"]:
            raise CommandError("pass --schemas A,B or --all")
        kw = snowflake_kwargs("SNOWFLAKE_ADMIN_" if o["admin"] else "SNOWFLAKE_")
        database = (o["database"] or kw.get("database") or tr.default_primary_database()).upper()
        kw["database"] = database
        try:
            sf = snowflake.connector.connect(**kw, login_timeout=30)
        except Exception as exc:
            raise CommandError(f"cannot connect to Snowflake: {exc}")
        db = Connection(database=database)
        syncer = Syncer(sf, db, database=database, batch_rows=o["batch_rows"], replace=o["replace"],
                        state_file=o["state_file"], log=lambda m: self.stdout.write(m))
        schemas = syncer.schemas() if o["all"] else [s.strip().upper() for s in o["schemas"].split(",") if s.strip()]
        only = [t.strip() for t in o["tables"].split(",") if t.strip()] or None
        failures = []
        for schema in schemas:
            self.stdout.write(self.style.MIGRATE_HEADING(f"== {database}.{schema}"))
            s = syncer.sync_schema(schema, only)
            self.stdout.write(f"   tables={s['tables']} views={s['views']} rows={s['rows']:,} "
                              f"skipped(already done)={s['skipped']} failed={len(s['failed'])}")
            failures += s["failed"]
        sf.close()
        db.close()
        if failures:
            for f in failures:
                self.stderr.write(self.style.ERROR(f"  ✗ {f}"))
            raise CommandError(f"{len(failures)} object(s) failed; re-run to retry them (completed ones are skipped)")
        self.stdout.write(self.style.SUCCESS("Done."))
