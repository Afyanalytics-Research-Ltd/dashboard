"""Install (or remove) the Warehouse Studio cost & performance toolkit in Snowflake.

    python manage.py warehouse_studio_install --env-file ~/secure/snowflake_admin.env
    python manage.py warehouse_studio_install --dry-run
    python manage.py warehouse_studio_install --uninstall --env-file ...
    python manage.py warehouse_studio_install --verify

The toolkit reads SNOWFLAKE.ACCOUNT_USAGE, so installing needs ACCOUNTADMIN.
Admin credentials are read from ``--env-file`` (or SNOWFLAKE_ADMIN_* variables)
and are used only for this command — the web app keeps its own, lower role.
"""

from __future__ import annotations

import os

from django.core.management.base import BaseCommand, CommandError

from warehouse_studio.conf import get_config
from warehouse_studio.services import installer
from warehouse_studio.services.connection import _load_private_key


def _read_env_file(path: str) -> dict[str, str]:
    values = {}
    with open(os.path.expanduser(path), encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            values[key.strip().removeprefix('export ').strip()] = value.strip().strip('"').strip("'")
    return values


class Command(BaseCommand):
    help = 'Install the Warehouse Studio cost & performance toolkit (procedures, views, task, grants) in Snowflake.'

    def add_arguments(self, parser):
        cfg = get_config()
        parser.add_argument('--env-file', help='File with SNOWFLAKE_ACCOUNT / USER / ROLE / PRIVATE_KEY_PATH or PASSWORD for an ACCOUNTADMIN login.')
        parser.add_argument('--database', default=cfg.toolkit_database)
        parser.add_argument('--schema', default=cfg.toolkit_schema)
        parser.add_argument('--app-role', default=(cfg.role or 'DATAANALYSTS'), help='Role the Studio connects with; receives USAGE/SELECT grants.')
        parser.add_argument('--task-warehouse', default=(cfg.warehouse or 'COMPUTE_WH'))
        parser.add_argument('--dry-run', action='store_true', help='Print the rendered SQL and exit.')
        parser.add_argument('--uninstall', action='store_true', help='Drop the toolkit (findings tables are kept unless --purge).')
        parser.add_argument('--purge', action='store_true', help='With --uninstall, also drop the findings tables.')
        parser.add_argument('--verify', action='store_true', help='Only report which toolkit objects exist.')

    def handle(self, *args, **opts):
        values = {'DB': opts['database'], 'SCHEMA': opts['schema'],
                  'APP_ROLE': opts['app_role'], 'TASK_WAREHOUSE': opts['task_warehouse']}
        name = 'uninstall.sql' if opts['uninstall'] else 'install.sql'
        try:
            steps = installer.load_steps(name, values)
        except installer.InstallError as exc:
            raise CommandError(str(exc)) from exc
        if opts['uninstall'] and not opts['purge']:
            steps = [s for s in steps if not s.sql.upper().startswith('DROP TABLE')]

        if opts['dry_run']:
            for step in steps:
                self.stdout.write(f'-- [{step.index}] {step.summary}\n{step.sql};\n')
            self.stdout.write(self.style.SUCCESS(f'{len(steps)} statements rendered (dry run).'))
            return

        conn = self._connect(opts.get('env_file'))
        try:
            if opts['verify']:
                report = installer.verify(conn, values['DB'].upper(), values['SCHEMA'].upper())
                for key, ok in {**report['procedures'], **report['relations']}.items():
                    self.stdout.write(f"  {'✔' if ok else '✘'} {key}")
                style = self.style.SUCCESS if report['installed'] else self.style.WARNING
                self.stdout.write(style('Toolkit installed.' if report['installed'] else f"Missing: {', '.join(report['missing'])}"))
                return

            def show(outcome):
                mark = self.style.SUCCESS('✔') if outcome['ok'] else self.style.ERROR('✘')
                self.stdout.write(f"{mark} [{outcome['step']:>3}] {outcome['summary']}")

            try:
                installer.apply(conn, steps, on_step=show)
            except installer.InstallError as exc:
                raise CommandError(str(exc)) from exc
            if not opts['uninstall']:
                report = installer.verify(conn, values['DB'].upper(), values['SCHEMA'].upper())
                if not report['installed']:
                    raise CommandError(f"Installed, but these objects are missing: {', '.join(report['missing'])}")
            self.stdout.write(self.style.SUCCESS(f"{'Removed' if opts['uninstall'] else 'Installed'} the toolkit in {values['DB']}.{values['SCHEMA']} ({len(steps)} statements)."))
        finally:
            conn.close()

    def _connect(self, env_file: str | None):
        import snowflake.connector

        from afya_backends import warehouse as afya_warehouse
        env = _read_env_file(env_file) if env_file else {}

        def get(key):
            # An --env-file is the complete credential set: never mix it with
            # ambient SNOWFLAKE_ADMIN_* values (that could pair one user with another's key).
            if env_file:
                return env.get(f'SNOWFLAKE_{key}', '')
            return os.getenv(f'SNOWFLAKE_ADMIN_{key}', '')

        account, user = get('ACCOUNT'), get('USER')
        if not account or not user:
            raise CommandError('Admin credentials missing: pass --env-file or set SNOWFLAKE_ADMIN_ACCOUNT / SNOWFLAKE_ADMIN_USER.')
        kwargs = {'account': account, 'user': user, 'role': get('ROLE') or 'ACCOUNTADMIN',
                  'application': 'AfyaWarehouseStudioInstaller'}
        if get('WAREHOUSE'):
            kwargs['warehouse'] = get('WAREHOUSE')
        key_path = get('PRIVATE_KEY_PATH')
        if key_path:
            kwargs['private_key'] = _load_private_key(os.path.expanduser(key_path), get('PRIVATE_KEY_PASSPHRASE'))
        elif get('PASSWORD'):
            kwargs['password'] = get('PASSWORD')
        else:
            raise CommandError('Admin credentials need SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD.')
        try:
            return afya_warehouse.connect(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f'Could not connect as {user}: {exc}') from exc
