import re
import shutil
import subprocess
import sys

from tests.conftest import INPUTS_DIR, OUTPUTS_DIR

# ↓ The FASTA header records the absolute checkpoint path, so the fixture carries
#   whichever machine minted it. Everything else in the header is reproducible.
_MODEL_PATH_FIELD = re.compile(r"model_path=\S*")


def _without_model_path(fasta: str) -> str:
    return _MODEL_PATH_FIELD.sub("model_path=<path>", fasta)


def test_protein_mpnn_output_unchanged(
    protein_mpnn_seed_111_1bc8_backbone, protein_mpnn_seed_111_1bc8_seq
):
    test_outputs_dir = OUTPUTS_DIR / "test_protein_mpnn_output_unchanged"

    if test_outputs_dir.exists():
        shutil.rmtree(test_outputs_dir)

    cmd = f"""{sys.executable} -m ligandmpnn --seed 111 --model_type protein_mpnn --pdb_path "{INPUTS_DIR!s}/1BC8.pdb" --out_folder "{test_outputs_dir!s}" """
    output = subprocess.run(cmd, shell=True, capture_output=True)

    assert output.returncode == 0, f"""Command: {cmd}
    Output:
    {output.stderr}"""

    # compare the outputs
    with open(test_outputs_dir / "backbones/1BC8_1.pdb") as f:
        backbone = f.read().strip()

    with open(test_outputs_dir / "seqs/1BC8.fa") as f:
        seq = f.read().strip()

    assert protein_mpnn_seed_111_1bc8_backbone == backbone
    assert _without_model_path(protein_mpnn_seed_111_1bc8_seq) == _without_model_path(
        seq
    )
