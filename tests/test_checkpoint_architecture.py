"""A checkpoint describes its own architecture.

Historically ``__main__.main`` read ``num_edges`` and ``atom_context_num`` from the
checkpoint but hardcoded the other five dimensions as literals, so any checkpoint
trained at a different width could not be loaded at all. These tests pin the
contract that replaces those literals: every dimension comes from the checkpoint,
and a checkpoint that omits them still builds the historical 128/3+3 network.
"""

import pytest
import torch

from ligandmpnn.models import (
    ProteinMPNN,
    build_model_from_checkpoint,
    model_kwargs_from_checkpoint,
)

# ↓ The shape of the shipped checkpoints: they carry num_edges/atom_context_num only.
LEGACY_CHECKPOINT_METADATA = {"num_edges": 32, "atom_context_num": 25}

# ↓ TMB_MPNN's shape (baker-laboratory/TMB_pipeline): a LigandMPNN trained wider.
TMB_CHECKPOINT_METADATA = {
    "num_edges": 96,
    "atom_context_num": 25,
    "node_features": 256,
    "edge_features": 256,
    "hidden_dim": 256,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_conditions": 4,
}


class TestModelKwargsFromCheckpoint:
    def test_a_legacy_checkpoint_still_builds_the_historical_network(self):
        """Absent keys must fall back, or all 19 shipped checkpoints stop loading."""
        kwargs = model_kwargs_from_checkpoint(LEGACY_CHECKPOINT_METADATA, "ligand_mpnn")
        assert kwargs["node_features"] == 128
        assert kwargs["edge_features"] == 128
        assert kwargs["hidden_dim"] == 128
        assert kwargs["num_encoder_layers"] == 3
        assert kwargs["num_decoder_layers"] == 3
        assert kwargs["k_neighbors"] == 32
        assert kwargs["atom_context_num"] == 25

    def test_a_wide_checkpoint_reports_its_own_dimensions(self):
        kwargs = model_kwargs_from_checkpoint(TMB_CHECKPOINT_METADATA, "ligand_mpnn")
        assert kwargs["node_features"] == 256
        assert kwargs["edge_features"] == 256
        assert kwargs["hidden_dim"] == 256
        assert kwargs["num_encoder_layers"] == 6
        assert kwargs["num_decoder_layers"] == 6
        assert kwargs["k_neighbors"] == 96
        assert kwargs["num_conditions"] == 4

    def test_only_the_ligand_family_reads_atom_context(self):
        """Upstream hardcodes atom_context_num=1 for every non-ligand model type."""
        kwargs = model_kwargs_from_checkpoint(
            LEGACY_CHECKPOINT_METADATA, "protein_mpnn"
        )
        assert kwargs["atom_context_num"] == 1

    def test_a_legacy_checkpoint_declares_no_conditions(self):
        """num_conditions must default to 0, or load_state_dict(strict=True) breaks."""
        kwargs = model_kwargs_from_checkpoint(LEGACY_CHECKPOINT_METADATA, "ligand_mpnn")
        assert kwargs["num_conditions"] == 0


class TestRoundTrip:
    """The real contract: a model's own checkpoint rebuilds it exactly.

    No test here can read a shipped ``.pt`` — ``ligandmpnn/model_params/`` ships
    empty — so the checkpoint is minted from a constructed model instead. That
    still exercises the thing that matters: whether the metadata we write is
    sufficient to reconstruct a network ``load_state_dict`` accepts strictly.
    """

    @staticmethod
    def _checkpoint_for(model: ProteinMPNN, **metadata) -> dict:
        return {"model_state_dict": model.state_dict(), **metadata}

    def test_a_narrow_model_round_trips(self):
        original = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32,
            atom_context_num=25,
            model_type="ligand_mpnn",
        )
        checkpoint = self._checkpoint_for(original, **LEGACY_CHECKPOINT_METADATA)
        rebuilt = build_model_from_checkpoint(checkpoint, "ligand_mpnn")
        assert rebuilt.hidden_dim == 128

    def test_a_wide_conditioned_model_round_trips_strictly(self):
        """The TMB case. A missing dimension surfaces here as a size mismatch."""
        original = ProteinMPNN(
            node_features=256,
            edge_features=256,
            hidden_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=6,
            k_neighbors=96,
            atom_context_num=25,
            model_type="ligand_mpnn",
            num_conditions=4,
        )
        checkpoint = self._checkpoint_for(original, **TMB_CHECKPOINT_METADATA)
        rebuilt = build_model_from_checkpoint(checkpoint, "ligand_mpnn")
        assert rebuilt.hidden_dim == 256
        assert rebuilt.condition_feat.weight.shape == (4, 256)

    def test_the_condition_head_is_absent_when_unconditioned(self):
        """A legacy model must not grow a head its checkpoint has no weights for."""
        original = ProteinMPNN(
            k_neighbors=32,
            atom_context_num=25,
            model_type="ligand_mpnn",
        )
        checkpoint = self._checkpoint_for(original, **LEGACY_CHECKPOINT_METADATA)
        rebuilt = build_model_from_checkpoint(checkpoint, "ligand_mpnn")
        assert getattr(rebuilt, "condition_feat", None) is None


class TestConditionHead:
    def test_an_unconditioned_model_has_no_condition_parameters(self):
        model = ProteinMPNN(model_type="ligand_mpnn", atom_context_num=25)
        assert "condition_feat.weight" not in dict(model.named_parameters())

    def test_a_conditioned_model_exposes_one_embedding_row_per_label(self):
        """0 unlabelled | 1 interface | 2 buried | 3 non-TMBB, per TMB_MPNN."""
        model = ProteinMPNN(
            model_type="ligand_mpnn", atom_context_num=25, num_conditions=4
        )
        assert model.condition_feat.weight.shape == (4, model.hidden_dim)

    def test_the_embedding_is_zero_initialised(self):
        """Matches upstream's nn.init.zeros_, so an untrained head is a no-op."""
        model = ProteinMPNN(
            model_type="ligand_mpnn", atom_context_num=25, num_conditions=4
        )
        assert torch.allclose(
            model.condition_feat.weight, torch.zeros_like(model.condition_feat.weight)
        )

    def test_requesting_labels_without_a_condition_head_is_refused(self):
        """Silently ignoring labels would look identical to conditioning working."""
        model = ProteinMPNN(model_type="ligand_mpnn", atom_context_num=25)
        with pytest.raises(ValueError, match="no condition head"):
            model.condition_embedding(torch.zeros(1, 10, dtype=torch.long))
