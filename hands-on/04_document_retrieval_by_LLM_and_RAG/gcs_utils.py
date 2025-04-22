from google.cloud import storage
from io import BytesIO, StringIO
from typing import BinaryIO
from logging import getLogger
import pandas as pd
import csv

logger = getLogger(__name__)

DOWNLOAD_CHUNK_SIZE = 419430400  # 400mb
UPLOAD_CHUNK_SIZE = 2097152  # 2mb
UPLOAD_TIMEOUT = 3600  # 1時間


class GcsUtils(object):
    def __init__(self, project_id: str, bucket_name: str):
        client = storage.Client(project_id)
        self.bucket = client.get_bucket(bucket_name)

    def read_csv_from_gcs(
        self,
        path: str,
        index=False,
        compression: str = 'infer'
    ) -> pd.DataFrame:
        # load dataset from gcs
        logger.info("read csv path prefix: {}".format(path))
        blobs = self.bucket.list_blobs(prefix=path)
        dataset = []
        for blob in blobs:
            blob.chunk_size = DOWNLOAD_CHUNK_SIZE
            content = blob.download_as_bytes(checksum=None)
            if index:
                tmp = pd.read_csv(BytesIO(content), compression=compression, index_col=0)
            else:
                tmp = pd.read_csv(BytesIO(content), compression=compression)
            dataset.append(tmp)
        return pd.concat([d for d in dataset])

    def create_file_readers_from_csv_files(self, path_prefix: str) -> list:
        readers = []
        logger.info("read readers path prefix: {}".format(path_prefix))
        blobs = self.bucket.list_blobs(prefix=path_prefix)
        for blob in blobs:
            blob = blob.download_as_string().decode('utf-8')
            blob = StringIO(blob)
            readers.append(csv.DictReader(blob))
        return readers

    def read_file_from_gcs(self, path: str) -> BinaryIO:
        logger.info("read file path: {}".format(path))
        blob = self.bucket.get_blob(path)
        data = blob.download_as_string()
        return BytesIO(data)

    def upload_file_to_gcs(self, gcs_path: str, local_path: str) -> None:
        logger.info("upload file path: {}".format(gcs_path))
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def upload_dataframe(self, path: str, df: pd.DataFrame, index=False) -> None:
        logger.info("upload dataframe path: {}".format(path))
        blob = self.bucket.blob(path, chunk_size=UPLOAD_CHUNK_SIZE)
        blob.upload_from_string(df.to_csv(index=index), timeout=UPLOAD_TIMEOUT)
