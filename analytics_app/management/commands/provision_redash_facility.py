"""
Provision (or update) a Redash Group + Data Source for one Facility.

Run snowflake/reporting_rls_setup.sql FIRST — this command only wires an
already-existing, facility-scoped Snowflake role/user into Redash. It does
not create anything in Snowflake itself. The Snowflake row access policy
remains the actual security boundary; this just confines a facility's
Redash users to a Data Source that can only see their own rows.

Usage:
    python manage.py provision_redash_facility \\
        --client wingspan --facility nairobi-west \\
        --snowflake-account UFLYZNZ-RA32706 \\
        --snowflake-user REPORTING_NAIROBI_WEST_SVC \\
        --snowflake-password '...' \\
        --snowflake-database HOSPITALS

    # Add an existing Redash user (matched by email) to the facility's group:
    python manage.py provision_redash_facility \\
        --client wingspan --facility nairobi-west \\
        --add-user someone@client.org --skip-datasource
"""

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.models import Client, Facility


class Command(BaseCommand):
    help = (
        'Provision a Redash Group + Data Source for one Facility, scoped to a '
        'facility-specific Snowflake role created by snowflake/reporting_rls_setup.sql.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--client', required=True, help='Client slug.')
        parser.add_argument('--facility', required=True, help='Facility slug (unique within the client).')
        parser.add_argument('--snowflake-account', help='Snowflake account identifier, e.g. UFLYZNZ-RA32706.')
        parser.add_argument('--snowflake-user', help='Facility-scoped Snowflake user created by the RLS setup script.')
        parser.add_argument('--snowflake-password', help='Password for the Snowflake user above.')
        parser.add_argument('--snowflake-warehouse', default='COMPUTE_WH')
        parser.add_argument('--snowflake-database', default='HOSPITALS')
        parser.add_argument('--add-user', help="Email of an existing Redash user to add to this facility's group.")
        parser.add_argument(
            '--skip-datasource', action='store_true',
            help='Only manage group / membership, skip data source creation.',
        )

    def handle(self, *args, **opts):
        base_url = settings.REDASH_BASE_URL
        api_key = settings.REDASH_ADMIN_API_KEY
        if not api_key:
            raise CommandError('REDASH_ADMIN_API_KEY is not set — add it to .env (Redash admin user -> Profile -> API Key).')

        session = requests.Session()
        session.headers.update({'Authorization': f'Key {api_key}'})

        try:
            client = Client.objects.get(slug=opts['client'])
        except Client.DoesNotExist:
            raise CommandError(f"No Client with slug={opts['client']!r}.")
        try:
            facility = Facility.objects.get(client=client, slug=opts['facility'])
        except Facility.DoesNotExist:
            raise CommandError(f"No Facility with slug={opts['facility']!r} under client {opts['client']!r}.")

        group_name = f'{client.slug}:{facility.slug}'
        group = self._get_or_create_group(session, base_url, group_name)
        self.stdout.write(self.style.SUCCESS(f"Group '{group_name}' -> id={group['id']}"))

        if not opts['skip_datasource']:
            required = ('snowflake_account', 'snowflake_user', 'snowflake_password')
            missing = [f'--{name.replace("_", "-")}' for name in required if not opts.get(name)]
            if missing:
                raise CommandError(f"Missing required options: {', '.join(missing)} (or pass --skip-datasource).")

            ds = self._get_or_create_datasource(session, base_url, facility, opts)
            self._attach_datasource_to_group(session, base_url, group['id'], ds['id'])
            self.stdout.write(self.style.SUCCESS(
                f"Data source '{ds['name']}' (id={ds['id']}) attached to group {group['id']}"
            ))

        if opts['add_user']:
            self._add_user_to_group(session, base_url, group['id'], opts['add_user'])
            self.stdout.write(self.style.SUCCESS(f"Added {opts['add_user']} to group {group['id']}"))

    # -- Redash API helpers ---------------------------------------------------

    def _get_or_create_group(self, session, base_url, name):
        resp = session.get(f'{base_url}/api/groups')
        resp.raise_for_status()
        for g in resp.json():
            if g['name'] == name:
                return g
        resp = session.post(f'{base_url}/api/groups', json={'name': name})
        resp.raise_for_status()
        return resp.json()

    def _get_or_create_datasource(self, session, base_url, facility, opts):
        # NOTE: 'schema' is not a real option for this connector (verified via
        # GET /api/data_sources/types — only account/region/user/password/
        # warehouse/database/host) — queries must fully-qualify schema.table.
        name = f'reporting-{facility.client.slug}-{facility.slug}'
        options = {
            'account': opts['snowflake_account'],
            'user': opts['snowflake_user'],
            'password': opts['snowflake_password'],
            'warehouse': opts['snowflake_warehouse'],
            'database': opts['snowflake_database'],
        }
        payload = {'name': name, 'type': 'snowflake', 'options': options}

        resp = session.get(f'{base_url}/api/data_sources')
        resp.raise_for_status()
        existing = next((ds for ds in resp.json() if ds['name'] == name), None)

        if existing:
            # Update in place — most commonly re-run to swap in a corrected
            # credential (e.g. after fixing a Snowflake auth policy) rather
            # than to change the name/type.
            resp = session.post(f"{base_url}/api/data_sources/{existing['id']}", json=payload)
        else:
            resp = session.post(f'{base_url}/api/data_sources', json=payload)

        if resp.status_code >= 400:
            raise CommandError(
                f'Redash rejected the data source ({resp.status_code}): {resp.text}\n'
                "The Snowflake connector's exact option keys can vary by Redash "
                'version — create one manually first via Redash Settings -> Data '
                'Sources -> New -> Snowflake to confirm field names, then adjust '
                'the `options` dict in this command to match.'
            )
        return resp.json()

    def _attach_datasource_to_group(self, session, base_url, group_id, ds_id):
        resp = session.post(
            f'{base_url}/api/groups/{group_id}/data_sources',
            json={'data_source_id': ds_id},
        )
        if resp.status_code >= 400 and 'already' not in resp.text.lower():
            resp.raise_for_status()

    def _add_user_to_group(self, session, base_url, group_id, email):
        resp = session.get(f'{base_url}/api/users', params={'q': email})
        resp.raise_for_status()
        body = resp.json()
        results = body.get('results', body) if isinstance(body, dict) else body
        user = next((u for u in results if u['email'].lower() == email.lower()), None)
        if not user:
            raise CommandError(
                f'No Redash user found for {email!r}. They must log into Redash '
                'at least once (or be invited via the Redash UI) before they can '
                'be added to a group.'
            )
        resp = session.post(f'{base_url}/api/groups/{group_id}/members', json={'user_id': user['id']})
        resp.raise_for_status()
