from pathlib import Path

from joblib import Parallel, delayed
from xyme_lake.lake import XymeDataLake

from ligandmpnn.logger import logger
from ligandmpnn.params.common import parameter_file_uris
from ligandmpnn.paths import MODEL_PARAMS_DIR

LAKE_MODEL_DIR = "s3://arn:aws:s3:eu-west-2:418272779708:accesspoint/dev-protein-science-access-point/public/protein-science/lmpnn_models"


def download_single_s3_file(lake: XymeDataLake, key: str, dest: Path):
    if dest.exists():
        logger.debug(f"Skipping existing file: {dest}")
        return

    lake.s3fs.get(
        lake.build_s3_uri(
            team_area="public", team="protein-science", key=f"lmpnn_models/{key}"
        ),
        str(dest),
    )

    logger.debug(f"Downloaded: {dest}")


def download_model_parameters(model_parameter_dir: Path = MODEL_PARAMS_DIR) -> None:
    """
    Download LigandMPNN and ProteinMPNN models from AWS S3.
    """

    def destination_path(object_key: str) -> Path:
        return model_parameter_dir / object_key

    object_keys = [Path(uri).name for uri in parameter_file_uris]
    lake = XymeDataLake()

    logger.info(f"Checking models from S3...")
    Parallel(n_jobs=2, prefer="threads")(
        delayed(download_single_s3_file)(lake=lake, key=key, dest=destination_path(key))
        for key in object_keys
    )
    logger.info(f"Models downloaded: {len(object_keys)}")
