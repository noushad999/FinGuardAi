-- FinGuard warehouse: schemas
--   raw       : source extracts loaded from the data lake (batch ETL)
--   ops       : operational tables written by streaming services (risk engine, Spark)
--   ml        : batch ML outputs (scores, segments, churn, graph, profiles)
--   analytics : dbt-managed staging views and star-schema marts (see dbt/)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS analytics;
