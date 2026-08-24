"""Unit tests for the backbone-only per-position distribution.

No checkpoint is loaded: a randomly initialised small model is enough to test
the *plumbing* properties (shape, normalisation, sequence-invariance), and it
keeps the test CPU-only and fast. Correctness of the learned weights is the
integration suite's job.
"""

import pytest
import torch

from ligandmpnn.data import featurize, parse_PDB
from ligandmpnn.models import ProteinMPNN
from tests.conftest import INPUTS_DIR


@pytest.fixture(scope="module")
def feature_dict() -> dict:
    """A real featurised backbone — NOT a hand-built dict.

    ↓ A stubbed feature_dict silently drops keys the model reads (chain_mask,
      randn, …) and the test then proves nothing about the real call.
    """
    protein_dict, _, _, _, _ = parse_PDB(
        str(INPUTS_DIR / "1BC8.pdb"), device=torch.device("cpu")
    )
    # ↓ featurize reads chain_mask unconditionally; parse_PDB doesn't set it —
    #   in the real CLI it's derived from --chains_to_design (score.py:168-185).
    #   All-ones designs every residue, which is what an unconditional prior wants.
    protein_dict["chain_mask"] = torch.ones_like(protein_dict["mask"])
    fd = featurize(protein_dict, model_type="protein_mpnn")
    fd["batch_size"] = 1
    return fd


@pytest.fixture(scope="module")
def model() -> ProteinMPNN:
    torch.manual_seed(0)
    net = ProteinMPNN(
        node_features=32,
        edge_features=32,
        hidden_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=8,
        model_type="protein_mpnn",
    )
    net.eval()
    return net


class TestUnconditionalProbs:
    def test_shape_is_positions_by_alphabet(self, model, feature_dict):
        out = model.unconditional_probs(feature_dict)
        length = feature_dict["S"].shape[1]
        assert out["log_probs"].shape == (1, length, 21)
        assert out["logits"].shape == (1, length, 21)

    def test_rows_are_normalised(self, model, feature_dict):
        out = model.unconditional_probs(feature_dict)
        row_sums = out["log_probs"].exp().sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_carries_no_decoding_order(self, model, feature_dict):
        # ↓ absence is the contract: score.py branches on it (Task 2).
        assert "decoding_order" not in model.unconditional_probs(feature_dict)

    def test_ignores_the_native_sequence(self, model, feature_dict):
        """The load-bearing property: unconditional means sequence-blind."""
        native = model.unconditional_probs(feature_dict)["logits"]
        scrambled_dict = dict(feature_dict)
        scrambled_dict["S"] = torch.randint_like(feature_dict["S"], low=0, high=20)
        scrambled = model.unconditional_probs(scrambled_dict)["logits"]
        assert torch.allclose(native, scrambled, atol=1e-6)
