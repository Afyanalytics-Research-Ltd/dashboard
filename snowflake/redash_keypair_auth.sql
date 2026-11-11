-- Switch Redash's Snowflake service user(s) to key-pair auth.
--
-- Why: Snowflake now enforces MFA on password logins for TYPE=PERSON users,
-- so Redash failed with "Multi-factor authentication is required for this
-- account". REDASH_SERVICE_AUTH_POLICY (password-only) no longer exempts it.
-- Redash's Snowflake runner is patched (redash/query_runner/snowflake.py) to
-- send a private key; the matching private key is ~/dashboard/rsa_key_redash.p8.
--
-- Run as ACCOUNTADMIN in Snowsight.

USE ROLE ACCOUNTADMIN;

-- Allow key-pair on the policy attached to the Redash service users.
ALTER AUTHENTICATION POLICY HOSPITALS.PUBLIC.REDASH_SERVICE_AUTH_POLICY
    SET AUTHENTICATION_METHODS = ('KEYPAIR', 'PASSWORD');

-- Mark as a non-human service user (exempt from MFA) and register the key.
ALTER USER REDASH_DEMO_SVC SET
    TYPE = SERVICE
    RSA_PUBLIC_KEY = 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuWWKqnngoYy9y57ScJXwNaGzYnHMUdKCfugjzAljHQ78HS88wTuaEk1MW3vIf3Ypw3lPjd7PGf5HaIl+ijJ2UfjKftGWARDS1Y1WEPP0ahfMrTx+KMyriBOSdqlXC+xj5nNtlmlYDK2Y2LNRUnGkTkZU9G7l9JOJaYPQDR3TkAMdXByEQbx7V/zLStPJQ0PvHLbA/LaoFL1reqwucy6k4rQPWmdsl3UB0vHBO26FVap9yv16QpIHfbnLqdvJtdIXSY8fW/lpCfP8xLy+gfwOygz2RueFenDlSlbOvYwXY3CTKnRq3sBQOpuWIzZ3hdsNypKiO36BFXf1XKnLjwhCdQIDAQAB';

DESC USER REDASH_DEMO_SVC;  -- expect TYPE=SERVICE and RSA_PUBLIC_KEY_FP set

-- Optional once Redash works: drop the password entirely.
-- ALTER USER REDASH_DEMO_SVC UNSET PASSWORD;
