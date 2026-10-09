"""Data-lake access (local filesystem or S3-compatible object storage).

``storage.lake_uri`` is either a local path (``data/lake``) or ``s3://bucket/prefix``.
S3 access goes through ``pyarrow.fs.S3FileSystem`` with an endpoint override so
the same code works against SeaweedFS/MinIO locally and AWS S3 in the cloud.

Lake zones::

    raw/<entity>/...                     immutable copies of source extracts (batch path)
    bronze/transactions/event_date=.../  validated, de-duplicated stream (Spark)
    quarantine/transactions/             records rejected by stream validation (Spark)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from common.config import PROJECT_ROOT, Settings, get_settings


class Lake:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        st = settings["storage"]
        uri = st["lake_uri"]
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            endpoint = urlparse(st["s3_endpoint"])
            self.fs: pafs.FileSystem = pafs.S3FileSystem(
                access_key=st["s3_access_key"],
                secret_key=st["s3_secret_key"],
                endpoint_override=endpoint.netloc,
                scheme=endpoint.scheme or "http",
                region=st.get("s3_region", "us-east-1"),
                allow_bucket_creation=True,
            )
            self.bucket = parsed.netloc
            self.root = f"{parsed.netloc}/{parsed.path.strip('/')}".rstrip("/")
            self.is_s3 = True
        else:
            p = Path(uri)
            p = p if p.is_absolute() else PROJECT_ROOT / p
            p.mkdir(parents=True, exist_ok=True)
            self.fs = pafs.LocalFileSystem()
            self.bucket = None
            self.root = str(p)
            self.is_s3 = False
        self.uri = uri

    def path(self, *parts: str) -> str:
        return "/".join([self.root, *[p.strip("/") for p in parts]])

    def ensure_bucket(self) -> None:
        if self.is_s3:
            self.fs.create_dir(self.bucket, recursive=False)

    def write_table(self, table: pa.Table, *parts: str, partition_cols: list[str] | None = None) -> str:
        """Write a table (overwriting the target) - idempotent re-runs."""
        target = self.path(*parts)
        if partition_cols:
            ds.write_dataset(
                table,
                target,
                filesystem=self.fs,
                format="parquet",
                partitioning=partition_cols,
                partitioning_flavor="hive",
                existing_data_behavior="delete_matching",
            )
        else:
            self.fs.create_dir(target.rsplit("/", 1)[0], recursive=True)
            pq.write_table(table, target, filesystem=self.fs)
        return target

    def read_table(self, *parts: str) -> pa.Table:
        return ds.dataset(self.path(*parts), filesystem=self.fs, format="parquet", partitioning="hive").to_table()

    def exists(self, *parts: str) -> bool:
        return self.fs.get_file_info(self.path(*parts)).type != pafs.FileType.NotFound
