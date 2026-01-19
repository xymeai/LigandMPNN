from ligandmpnn.params import download_models
from ligandmpnn.paths import MODEL_PARAMS_DIR


def test_download_models():
    download_models()
    param_files = list(MODEL_PARAMS_DIR.glob("*.pt"))

    assert len(param_files) == 15
