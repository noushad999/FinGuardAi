import os

import pytest

if os.environ.get("FINGUARD_INTEGRATION") != "1":
    pytest.skip("integration tests need the running stack (FINGUARD_INTEGRATION=1)", allow_module_level=True)

API = os.environ.get("API_URL", "http://localhost:8000")
DB = os.environ.get("DATABASE_URL", "postgresql+psycopg2://finguard:finguard@localhost:5432/finguard")
KAFKA = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
S3 = os.environ.get("S3_ENDPOINT", "http://localhost:8333")


@pytest.fixture(scope="session")
def engine():
    from sqlalchemy import create_engine

    return create_engine(DB)
