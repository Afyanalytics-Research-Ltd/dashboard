"""
Snowflake connection — shared across all analysis modules.
Fill in credentials below after cloning. This file is gitignored; never commit real values.
"""
import snowflake.connector

SF_USER      = 
SF_PASSWORD  = 
SF_ACCOUNT   = 
SF_DATABASE  = 
SF_SCHEMA    = 
SF_WAREHOUSE = 


def connect(passcode: str):
    return snowflake.connector.connect(
        user=SF_USER,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        warehouse=SF_WAREHOUSE,
        authenticator="username_password_mfa",
        passcode=passcode,
    )
