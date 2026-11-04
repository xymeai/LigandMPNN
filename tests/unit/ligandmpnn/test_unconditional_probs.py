"""Unit tests for the backbone-only per-position distribution.

No checkpoint weights are trained: a randomly initialised small model is enough to
test the *plumbing* properties (shape, normalisation, sequence- and order-invariance),
and it keeps the tests CPU-only and fast. Correctness of the learned weights is the
integration suite's job.
"""

import pytest
import torch

from ligandmpnn import score
from ligandmpnn.data import featurize, parse_PDB
from ligandmpnn.models import ProteinMPNN
from tests.conftest import INPUTS_DIR

LIGAND_PDB = "4GYT.pdb"


def _featurise(model_type: str, pdb: str, chain_mask_zero_for: int = 0) -> dict:
    """A real featurised backbone — NOT a hand-built dict.

    ↓ A stubbed feature_dict silently drops keys the model reads (chain_mask,
      randn, …) and the test then proves nothing about the real call.
    ``chain_mask_zero_for`` fixes (mask 0) the first N residues, as
    --fixed_residues would.
    """
    protein_dict, _, _, _, _ = parse_PDB(
        str(INPUTS_DIR / pdb),
        device=torch.device("cpu"),
        parse_all_atoms=model_type == "ligand_mpnn",
    )
    # ↓ featurize reads chain_mask unconditionally; parse_PDB doesn't set it —
    #   in the real CLI it's derived from --chains_to_design (score.py:164-185).
    chain_mask = torch.ones_like(protein_dict["mask"])
    chain_mask[:chain_mask_zero_for] = 0
    protein_dict["chain_mask"] = chain_mask
    fd = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=1,
        number_of_ligand_atoms=16,
        model_type=model_type,
    )
    fd["batch_size"] = 1
    fd["randn"] = torch.randn([1, fd["mask"].shape[1]])
    fd["symmetry_residues"] = [[]]  # score() expects a list of lists
    return fd


def _tiny_model(model_type: str, side_chain_context: bool = False) -> ProteinMPNN:
    torch.manual_seed(0)
    net = ProteinMPNN(
        node_features=32,
        edge_features=32,
        hidden_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=8,
        atom_context_num=16 if model_type == "ligand_mpnn" else 0,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=side_chain_context,
    )
    net.eval()
    return net


@pytest.fixture(scope="module")
def feature_dict() -> dict:
    return _featurise("protein_mpnn", "1BC8.pdb")


@pytest.fixture(scope="module")
def model() -> ProteinMPNN:
    return _tiny_model("protein_mpnn")


@pytest.fixture(scope="module")
def ligand_feature_dict() -> dict:
    return _featurise("ligand_mpnn", LIGAND_PDB)


@pytest.fixture(scope="module")
def ligand_model() -> ProteinMPNN:
    return _tiny_model("ligand_mpnn")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A 128-dim random checkpoint in the shape score.py loads."""
    torch.manual_seed(0)
    net = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=8,
        model_type="protein_mpnn",
    )
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model_state_dict": net.state_dict(), "num_edges": 8}, path)
    return str(path)


def _score_args(tmp_path, checkpoint: str, *extra: str):
    return score.build_argparser().parse_args(
        [
            "--model_type",
            "protein_mpnn",
            "--checkpoint_protein_mpnn",
            checkpoint,
            "--pdb_path",
            str(INPUTS_DIR / "1BC8.pdb"),
            "--out_folder",
            str(tmp_path) + "/",
            "--seed",
            "111",
            "--verbose",
            "0",
            *extra,
        ]
    )


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
        # ↓ absence is the contract: score.py branches on it.
        assert "decoding_order" not in model.unconditional_probs(feature_dict)

    def test_ignores_the_native_sequence(self, model, feature_dict):
        """Sequence-blind, including the X token (index 20)."""
        native = model.unconditional_probs(feature_dict)["logits"]
        scrambled_dict = dict(feature_dict)
        scrambled_dict["S"] = torch.randint_like(feature_dict["S"], low=0, high=21)
        scrambled = model.unconditional_probs(scrambled_dict)["logits"]
        assert torch.allclose(native, scrambled, atol=1e-6)

    def test_is_independent_of_decoding_order(self, model, feature_dict):
        """The property that makes it a prior: a different randn must change nothing."""
        first = model.unconditional_probs(feature_dict)["logits"]
        reshuffled = dict(feature_dict)
        reshuffled["randn"] = torch.randn_like(feature_dict["randn"])
        assert torch.allclose(first, model.unconditional_probs(reshuffled)["logits"])

    def test_equals_single_aa_score_with_use_sequence_true(self, model, feature_dict):
        """One pass reproduces the L-pass mode whose flag name says the opposite.

        ↓ In single_aa_score the scored position is decoded FIRST when
          use_sequence=True, so it sees no neighbour residue; that is exactly
          the unconditional quantity, at L decoder passes instead of one.
        """
        prior = model.unconditional_probs(feature_dict)["logits"]
        looped = model.single_aa_score(feature_dict, use_sequence=True)["logits"]
        assert torch.allclose(prior, looped, atol=1e-5)

    def test_differs_from_the_full_context_and_autoregressive_modes(
        self, model, feature_dict
    ):
        """use_sequence=False decodes the scored position LAST (sees all neighbours)."""
        prior = model.unconditional_probs(feature_dict)["logits"]
        for other in (
            model.single_aa_score(feature_dict, use_sequence=False)["logits"],
            model.score(feature_dict, use_sequence=True)["logits"],
            model.score(feature_dict, use_sequence=False)["logits"],
        ):
            assert not torch.allclose(prior, other, atol=1e-4)

    def test_rejects_a_batched_encoder_input(self, model, feature_dict):
        batched = dict(feature_dict)
        batched["S"] = feature_dict["S"].repeat(2, 1)
        with pytest.raises(ValueError, match="batch"):
            model.unconditional_probs(batched)


class TestLigandMPNN:
    """ligand_mpnn is the Xyme model type; it has its own encoder branch."""

    def test_shape_normalisation_and_sequence_invariance(
        self, ligand_model, ligand_feature_dict
    ):
        out = ligand_model.unconditional_probs(ligand_feature_dict)
        length = ligand_feature_dict["S"].shape[1]
        assert out["logits"].shape == (1, length, 21)
        row_sums = out["log_probs"].exp().sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        scrambled = dict(ligand_feature_dict)
        scrambled["S"] = torch.randint_like(ligand_feature_dict["S"], low=0, high=21)
        assert torch.allclose(
            out["logits"],
            ligand_model.unconditional_probs(scrambled)["logits"],
            atol=1e-6,
        )


class TestSideChainLeakGuard:
    """Side-chain atoms of fixed residues feed the ligand_mpnn encoder.

    ↓ Side chains are near-perfect residue-identity labels, so a 'backbone-only'
      prior computed with them is conditioned on the native sequence. Refuse
      rather than stamp mode='unconditional' on it.
    """

    def test_refuses_when_fixed_residues_carry_side_chain_context(self):
        net = _tiny_model("ligand_mpnn", side_chain_context=True)
        fd = _featurise("ligand_mpnn", LIGAND_PDB, chain_mask_zero_for=5)
        with pytest.raises(ValueError, match="side-chain"):
            net.unconditional_probs(fd)

    def test_allows_side_chain_context_when_nothing_is_fixed(self):
        net = _tiny_model("ligand_mpnn", side_chain_context=True)
        fd = _featurise("ligand_mpnn", LIGAND_PDB)
        assert net.unconditional_probs(fd)["logits"].shape[-1] == 21

    def test_allows_fixed_residues_without_side_chain_context(self):
        net = _tiny_model("ligand_mpnn", side_chain_context=False)
        fd = _featurise("ligand_mpnn", LIGAND_PDB, chain_mask_zero_for=5)
        assert net.unconditional_probs(fd)["logits"].shape[-1] == 21


class TestScoreArgsSurface:
    """The CLI contract xyme-tools builds a Namespace against."""

    def test_parser_exposes_unconditional_probs_only(self):
        args = score.build_argparser().parse_args([])
        assert args.unconditional_probs_only == 0

    def test_flag_is_accepted(self):
        args = score.build_argparser().parse_args(["--unconditional_probs_only", "1"])
        assert args.unconditional_probs_only == 1


class TestScoreFlagGuards:
    """Flags that the unconditional mode would silently ignore must fail loudly."""

    @pytest.mark.parametrize(
        "flags",
        [
            ["--autoregressive_score", "1"],
            ["--fixed_residues", "A1 A2"],
            ["--redesigned_residues", "A1 A2"],
            ["--symmetry_residues", "A1,A2|A3,A4"],
            ["--homo_oligomer", "1"],
            ["--chains_to_design", "A"],
        ],
    )
    def test_incompatible_flag_exits_before_any_work(
        self, tmp_path, tiny_checkpoint, flags
    ):
        args = _score_args(
            tmp_path, tiny_checkpoint, "--unconditional_probs_only", "1", *flags
        )
        with pytest.raises(SystemExit) as excinfo:
            score.main(args)
        assert "unconditional_probs_only" in str(excinfo.value.code)
        assert not list(tmp_path.glob("1BC8*.pt"))


class TestScoreMainUnconditional:
    def test_writes_a_self_describing_pt(self, tmp_path, tiny_checkpoint):
        """Runs the real score.main against a randomly initialised checkpoint.

        ↓ Only --unconditional_probs_only is set — the single-flag path a real
          caller would use, now that it is tested first in the mode chain.
        """
        score.main(
            _score_args(tmp_path, tiny_checkpoint, "--unconditional_probs_only", "1")
        )
        loaded = torch.load(next(tmp_path.glob("1BC8*.pt")), weights_only=False)
        assert loaded["mode"] == "unconditional"
        assert "decoding_order" not in loaded
        assert len(loaded["alphabet"]) == loaded["probs"].shape[-1] == 21
        assert loaded["probs"].shape[1] == len(loaded["residue_names"])
        # ↓ a valid distribution, not just the right shape — catches a
        #   miswired forward pass that shape/key checks alone would miss.
        assert (abs(loaded["probs"].sum(axis=-1) - 1.0) < 1e-5).all()

    def test_says_what_post_processing_was_not_applied(self, tmp_path, tiny_checkpoint):
        score.main(
            _score_args(tmp_path, tiny_checkpoint, "--unconditional_probs_only", "1")
        )
        loaded = torch.load(next(tmp_path.glob("1BC8*.pt")), weights_only=False)
        assert "X" in loaded["post_processing"]
        assert "bias" in loaded["post_processing"]

    def test_batching_is_collapsed_to_one_deterministic_pass(
        self, tmp_path, tiny_checkpoint
    ):
        """N identical rows would make std_of_probs a fabricated zero."""
        score.main(
            _score_args(
                tmp_path,
                tiny_checkpoint,
                "--unconditional_probs_only",
                "1",
                "--batch_size",
                "2",
                "--number_of_batches",
                "3",
            )
        )
        loaded = torch.load(next(tmp_path.glob("1BC8*.pt")), weights_only=False)
        assert loaded["probs"].shape[0] == 1
        assert "std_of_probs" not in loaded

    def test_namespace_built_before_the_flag_existed_still_works(
        self, tmp_path, tiny_checkpoint
    ):
        """xyme-tools hand-builds Namespaces; an upgrade must not raise."""
        args = _score_args(tmp_path, tiny_checkpoint)
        del args.unconditional_probs_only
        score.main(args)
        loaded = torch.load(next(tmp_path.glob("1BC8*.pt")), weights_only=False)
        assert loaded["mode"] == "single_aa"
        assert "std_of_probs" in loaded
