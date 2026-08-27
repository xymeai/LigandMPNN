"""Non-canonical residues must be decodable positions with their canonical identity.

Two independent silent failures motivate these tests, both measured on a real 7-state enzyme
ensemble (Xyme lipase cycle 7, PROTDES-638):

* A residue name outside ProDy's amino-acid set is not `protein`, so it contributes no CA and
  DISAPPEARS from the decode — and its atoms, being "not protein and not water", are handed to
  the model as LIGAND context instead. Measured: renaming one SER to SED took the ensemble from
  1267 positions / 95 ligand atoms to 1266 / 101, and the caller's tied-decode indices (built
  from Rosetta's residue count, which does count SED) then ran off the end:
  `IndexError: index 1266 is out of bounds for dimension 0 with size 1266`.
* A residue name outside `restype_3to1` decodes as `X`. ProDy knows HIP, so it survives as a
  position — but it reached the model as UNKNOWN rather than histidine, at all four catalytic
  HIP156 sites. Every protonation variant the enzyme-design pipelines rely on (ASH, GLH, HIP,
  SED) was affected.

Neither mode logs anything: `parse_PDB` has no warning path and no strict mode.
"""

import numpy as np
import pytest
from prody import flagDefinition

from ligandmpnn.data import NONCANONICAL_3TO1, parse_PDB

UNKNOWN_TOKEN = 20

_CANONICAL_1TO3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# ↓ Backbone-complete residues; only N/CA/C/O matter to the parser, so sidechain atoms are
#   omitted deliberately — the tests are about NAMES, and a sidechain would imply otherwise.
_TEMPLATE = [
    ("N", 0.000, 0.000, 0.000), ("CA", 1.458, 0.000, 0.000),
    ("C", 2.009, 1.420, 0.000), ("O", 1.251, 2.390, 0.000),
]


def _pdb(tmp_path, names: list[str]) -> str:
    """One chain of backbone-only residues with the given names, 3.8 A apart along x."""
    lines, serial = [], 1
    for i, name in enumerate(names, start=1):
        for atom, x, y, z in _TEMPLATE:
            # ↓ Columns matter: serial 7-11, atom name 13-16, altLoc 17, resName 18-20,
            #   chain 22, resSeq 23-26, coords 31-54. Getting resName one column early makes
            #   ProDy silently skip the line, which reads as a parser bug rather than a bad
            #   fixture.
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {name:>3s} A{i:4d}    "
                f"{x + 3.8 * i:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{atom[0]:>2s}"
            )
            serial += 1
    path = tmp_path / "res.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


class TestEveryAliasIsADecodablePosition:
    """The drop is the dangerous mode: the position vanishes AND becomes ligand context."""

    @pytest.mark.parametrize("resname", sorted(NONCANONICAL_3TO1))
    def test_prody_treats_it_as_protein(self, resname: str) -> None:
        assert resname in flagDefinition("protein"), (
            f"{resname} is not in ProDy's amino-acid set, so it contributes no CA atom and is "
            "silently dropped from the decode while its atoms become ligand context."
        )

    @pytest.mark.parametrize("resname", sorted(NONCANONICAL_3TO1))
    def test_it_survives_parsing_as_one_position(self, tmp_path, resname: str) -> None:
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"]))[0]
        assert len(parsed["S"]) == 3, f"{resname} cost a position: {len(parsed['S'])} != 3"

    @pytest.mark.parametrize("resname", sorted(NONCANONICAL_3TO1))
    def test_it_does_not_leak_into_ligand_context(self, tmp_path, resname: str) -> None:
        """A dropped residue does not merely vanish — it reappears as SUBSTRATE.

        Compared against the canonical parent rather than against zero: ``Y_t`` is never
        empty (an all-ALA file already yields one entry), so an absolute assertion would
        fail on correct input and prove nothing. The property that matters is that the
        alias adds no ligand atoms its parent would not — measured on the real ensemble as
        95 atoms with SER against 101 with SED, i.e. SED's six heavy atoms.
        """
        alias = np.asarray(parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"]))[0]["Y_t"])
        parent3 = {v: k for k, v in _CANONICAL_1TO3.items()}[NONCANONICAL_3TO1[resname]]
        parent = np.asarray(parse_PDB(_pdb(tmp_path, ["ALA", parent3, "ALA"]))[0]["Y_t"])
        assert alias.shape[0] <= parent.shape[0], (
            f"{resname} contributed {alias.shape[0] - parent.shape[0]} more ligand-context "
            f"atom(s) than its parent {parent3}: it was classified as 'not protein and not "
            "water' and handed to the model as part of the substrate."
        )


class TestEveryAliasDecodesAsItsCanonicalParent:
    @pytest.mark.parametrize("resname,expected", sorted(NONCANONICAL_3TO1.items()))
    def test_identity_is_the_parent_not_unknown(
        self, tmp_path, resname: str, expected: str
    ) -> None:
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"]))[0]
        S = np.asarray(parsed["S"])
        assert S[1] != UNKNOWN_TOKEN, f"{resname} decoded as X, not {expected}"
        parent3 = {v: k for k, v in _CANONICAL_1TO3.items()}[expected]
        reference = np.asarray(parse_PDB(_pdb(tmp_path, ["ALA", parent3, "ALA"]))[0]["S"])
        assert S[1] == reference[1], (
            f"{resname} decoded as token {S[1]}, but its parent {expected} ({parent3}) "
            f"is token {reference[1]}"
        )


class TestThePanelTheDesignPipelinesActuallyUse:
    """ASH / GLH / HIP / SED are the assembled Xyme variant panel; all four must work."""

    @pytest.mark.parametrize("resname", ["ASH", "GLH", "HIP", "SED"])
    def test_panel_member_is_covered(self, resname: str) -> None:
        assert resname in NONCANONICAL_3TO1, f"{resname} is in the design panel but unmapped"
