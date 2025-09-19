from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
FIXTURE_DIR = TEST_DIR / "fixtures"
OUTPUTS_DIR = TEST_DIR / "outputs"
INPUTS_DIR = REPO_DIR / "inputs"

@pytest.fixture(scope="session")
def protein_mpnn_seed_111_1bc8_backbone() -> str:
    with open(FIXTURE_DIR / "protein_mpnn_seed-111_1BC8/backbones/1BC8_1.pdb") as f:
        return f.read().strip()

@pytest.fixture(scope="session")
def protein_mpnn_seed_111_1bc8_seq() -> str:
    with open(FIXTURE_DIR / "protein_mpnn_seed-111_1BC8/seqs/1BC8.fa") as f:
        return f.read().strip()