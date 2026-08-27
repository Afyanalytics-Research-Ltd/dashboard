

-- ============================================================================
-- Facility-scoped row access policy for HOSPITALS.REPORTING
--
-- Run this by hand in a Snowflake worksheet (as a role with SECURITYADMIN /
-- CREATE ROW ACCESS POLICY / MANAGE GRANTS privileges). It is NOT a dbt
-- model — it defines security objects (roles, policies), not transformations.
--
-- This is the actual enforcement point for "users can only see their own
-- facility's data". Everything on the Redash/Django side (Groups, Data
-- Sources, the `p_facility` embed parameter) is UX and defense-in-depth on
-- top of this — someone who reaches the underlying Snowflake role directly
-- is still bound by the policy below.
-- ============================================================================

USE ROLE SECURITYADMIN;

CREATE SCHEMA IF NOT EXISTS HOSPITALS.REPORTING;

-- ----------------------------------------------------------------------------
-- 0. Auth exemption for Redash's service accounts.
--
--    This account requires key-pair (or SSO/MFA) authentication for
--    interactive users. Redash's built-in Snowflake connector only supports
--    plain username + password (confirmed via this Redash build's own
--    GET /api/data_sources/types — no private-key field exists), so every
--    per-facility service user Redash connects as (step 4 below) needs to be
--    explicitly exempted from that account-wide requirement. Everyone else
--    keeps key-pair/SSO — this policy is attached per-user, never at the
--    account level.
-- ----------------------------------------------------------------------------
CREATE AUTHENTICATION POLICY IF NOT EXISTS HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY
    AUTHENTICATION_METHODS = ('PASSWORD')
    COMMENT = 'Allows password auth for non-interactive BI service accounts (Redash) that cannot do key-pair.';
-- Attach with, per service user created below:
--   ALTER USER <facility>_SVC SET AUTHENTICATION POLICY HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY;
--
-- If your account's global authentication policy is enforced in a way that
-- doesn't allow per-user override (e.g. mandated centrally via your IdP with
-- no exceptions), this statement will be rejected — in that case this whole
-- approach is a dead end and Redash needs either a newer/patched connector
-- with key-pair support, or a replicated copy of REPORTING in a warehouse
-- Redash can reach with key-pair-free auth.

-- ----------------------------------------------------------------------------
-- 1. Role -> facility mapping table.
--    One row per (facility-scoped role, facility_code) it may see.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS HOSPITALS.REPORTING.FACILITY_ROLE_MAP (
    role_name     VARCHAR NOT NULL,
    facility_code VARCHAR NOT NULL,
    PRIMARY KEY (role_name, facility_code)
);

-- Roles listed here bypass the facility filter entirely — this is the
-- Snowflake-side equivalent of the client_admin / facilities_admin tier.
-- Keep this list short and reviewed; anyone connecting as one of these
-- roles sees every facility's rows.
CREATE TABLE IF NOT EXISTS HOSPITALS.REPORTING.UNRESTRICTED_ROLES (
    role_name VARCHAR NOT NULL PRIMARY KEY
);
-- Example:
-- INSERT INTO HOSPITALS.REPORTING.UNRESTRICTED_ROLES (role_name) VALUES ('REPORTING_CLIENT_ADMIN_ROLE');

-- ----------------------------------------------------------------------------
-- 2. The row access policy itself.
--    Attach to any REPORTING table/view that carries a FACILITY_CODE column.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE ROW ACCESS POLICY HOSPITALS.REPORTING.FACILITY_RLS_POLICY
AS (facility_code VARCHAR) RETURNS BOOLEAN ->
    EXISTS (
        SELECT 1 FROM HOSPITALS.REPORTING.UNRESTRICTED_ROLES u
        WHERE u.role_name = CURRENT_ROLE()
    )
    OR EXISTS (
        SELECT 1 FROM HOSPITALS.REPORTING.FACILITY_ROLE_MAP m
        WHERE m.role_name = CURRENT_ROLE()
          AND m.facility_code = facility_code
    );

-- ----------------------------------------------------------------------------
-- 3. Attach the policy — repeat per reporting table/view.
-- ----------------------------------------------------------------------------
-- ALTER TABLE HOSPITALS.REPORTING.SOME_FACT_TABLE
--     ADD ROW ACCESS POLICY HOSPITALS.REPORTING.FACILITY_RLS_POLICY ON (facility_code);

-- ----------------------------------------------------------------------------
-- 4. Pattern: one scoped role + service user per facility.
--    Repeat this block per facility (swap NAIROBI_WEST for the real code
--    used in core.Facility.slug, upper-cased, e.g. `nairobi-west` -> `NAIROBI_WEST`).
-- ----------------------------------------------------------------------------
USE ROLE SECURITYADMIN;
CREATE ROLE IF NOT EXISTS APPRENTICE_ROLE;
GRANT USAGE ON WAREHOUSE COMPUTE_WH               TO ROLE APPRENTICE_ROLE;
GRANT USAGE ON DATABASE HOSPITALS                 TO ROLE APPRENTICE_ROLE;
GRANT USAGE ON SCHEMA HOSPITALS.REPORTING         TO ROLE APPRENTICE_ROLE;
GRANT SELECT ON ALL TABLES IN SCHEMA HOSPITALS.REPORTING    TO ROLE APPRENTICE_ROLE;
GRANT SELECT ON FUTURE TABLES IN SCHEMA HOSPITALS.REPORTING TO ROLE APPRENTICE_ROLE;
--
CREATE USER IF NOT EXISTS APPRENTICE_SVC
    PASSWORD = 'STRONGPASSWORD!'
    DEFAULT_ROLE = APPRENTICE_ROLE
    DEFAULT_WAREHOUSE = COMPUTE_WH
    MUST_CHANGE_PASSWORD = FALSE;
GRANT ROLE APPRENTICE_ROLE TO USER APPRENTICE_SVC;

-- Required so Redash (password-only) can actually log in as this user —
-- see section 0 above.
ALTER USER APPRENTICE_SVC
    SET AUTHENTICATION POLICY HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY;

INSERT INTO HOSPITALS.REPORTING.FACILITY_ROLE_MAP (role_name, facility_code)
VALUES ('APPRENTICE_ROLE', 'KISUMU');

-- The REPORTING_NAIROBI_WEST_SVC / password pair above is what you then feed
-- into `provision_redash_facility` (see
-- analytics_app/management/commands/provision_redash_facility.py) as
-- --snowflake-user / --snowflake-password, so Redash's Data Source for that
-- facility connects as this exact scoped role.
--
-- ----------------------------------------------------------------------------
-- 5. Unblock-the-demo-today shortcut: a single service user on the existing
--    DATAANALYSTS role, with no facility filtering yet (RLS not attached to
--    anything — steps 1-3 above are what add real facility scoping later).
--    Swap this out for the per-facility version above once ready.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE USER REDASH_DEMO_SVC
    PASSWORD = 'STRONGPASSWORD!'
    DEFAULT_ROLE = DATAANALYSTS
    DEFAULT_WAREHOUSE = COMPUTE_WH
    MUST_CHANGE_PASSWORD = FALSE;
GRANT ROLE DATAANALYSTS TO USER REDASH_DEMO_SVC;
ALTER USER REDASH_DEMO_SVC SET AUTHENTICATION POLICY HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY;
ALTER AUTHENTICATION POLICY HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY
    SET MFA_ENROLLMENT = 'OPTIONAL';