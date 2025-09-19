import shutil
import subprocess
import sys

from tests.conftest import OUTPUTS_DIR, INPUTS_DIR


def test_protein_mpnn_output_unchanged(protein_mpnn_seed_111_1bc8_backbone, protein_mpnn_seed_111_1bc8_seq):
    test_outputs_dir = OUTPUTS_DIR / 'test_protein_mpnn_output_unchanged'

    if test_outputs_dir.exists():
        shutil.rmtree(test_outputs_dir)

    cmd = f"""{sys.executable} -m ligandmpnn --seed 111 --model_type protein_mpnn --pdb_path "{str(INPUTS_DIR)}/1BC8.pdb" --out_folder "{str(test_outputs_dir)}" """
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
    assert protein_mpnn_seed_111_1bc8_seq == seq