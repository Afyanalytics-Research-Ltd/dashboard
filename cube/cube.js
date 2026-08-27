const SnowflakeDriver = require('@cubejs-backend/snowflake-driver');
const fs = require('fs');

module.exports = {
  driverFactory: () =>
    new SnowflakeDriver({
      account: 'MW76455.eu-north-1.aws',
      username: process.env.CUBEJS_DB_USER,
      warehouse: process.env.CUBEJS_DB_WAREHOUSE,
      database: process.env.CUBEJS_DB_NAME,
      schema: process.env.CUBEJS_DB_SCHEMA,
      role: process.env.CUBEJS_DB_SNOWFLAKE_ROLE,
      authenticator: 'SNOWFLAKE_JWT',
      privateKey: fs.readFileSync(process.env.CUBEJS_DB_SNOWFLAKE_PRIVATE_KEY_PATH, 'utf8'),
    }),
};
