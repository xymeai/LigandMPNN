"""Non-canonical residues: decodable positions, canonical identity, and no lost chemistry.

Three properties, measured on a real seven-state enzyme ensemble before being pinned here:

* A name ProDy does not know contributes no CA atom, so the position DISAPPEARS and its atoms
  are handed to the model as ligand context. Renaming one SER to SED took the ensemble from
  1267 positions to 1266, and the caller's tied indices — built from Rosetta's count, which
  does count SED — then ran off the end of the tensor.
* A name outside ``restype_3to1`` decodes as ``X``. ProDy knows HIP, so it survived as a
  position but reached the model as UNKNOWN at all four catalytic HIP156 sites.
* Registering a STRUCTURALLY MODIFIED residue is the opposite mistake. Its extra atoms are in
  neither the 37 named slots nor ligand context, so they vanish entirely: an LLP-shaped
  residue (PLP-lysine) contributes 16 ligand-context atoms unregistered and 1 registered. The
  tier split exists for that reason, and the tier-B test below is what keeps it honest.
"""

import numpy as np
import pytest
from prody import flagDefinition

from ligandmpnn.data import (
    MODIFIED_RESIDUE_3TO1,
    NONCANONICAL_3TO1,
    PROTONATION_VARIANT_3TO1,
    parse_PDB,
)

UNKNOWN_TOKEN = 20

# ↓ Captured from ProDy's own table. Import order matters: ligandmpnn.data registers tier A on
#   import, so this must be read AFTER that, and it deliberately reflects ProDy's defaults plus
#   whatever the user's settings carry — the point is "not registered BY US".
_PRODY_NATIVE = frozenset(flagDefinition("protein")) - frozenset(PROTONATION_VARIANT_3TO1)
_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_BACKBONE = [("N", 0.0, 0.0, 0.0), ("CA", 1.458, 0.0, 0.0),
             ("C", 2.009, 1.420, 0.0), ("O", 1.251, 2.390, 0.0)]
# ↓ a side chain large enough for the ligand-context test to mean something: on a
#   backbone-only residue an alias and its parent yield identical Y_t, so the assertion
#   would compare two equal constants and pass whatever the code did.
_SIDECHAIN = [("CB", 2.5, -1.0, 0.0), ("CG", 3.5, -1.5, 0.0), ("CD", 4.5, -2.0, 0.0),
              ("CE", 5.5, -2.5, 0.0), ("NZ", 6.5, -3.0, 0.0)]


def _pdb(tmp_path, names: list[str], sidechain: bool = False, hetatm_idx: int | None = None) -> str:
    """One chain of residues with the given names, 3.8 A apart along x.

    Columns are explicit because getting them wrong makes ProDy skip or mis-read the line,
    which reads as a parser bug rather than a bad fixture: serial 7-11, atom name 13-16,
    altLoc 17, resName 18-20 (a 4-character name legitimately spills into 21), chain 22,
    resSeq 23-26, coordinates 31-54. ``resName`` is therefore LEFT-justified in 18-21 after an
    explicit altLoc space, so a four-character name (HISD/HISE/HISP) occupies 18-21 and a
    three-character one occupies 18-20 leaving 21 blank. Right-justifying it instead puts a
    four-character name in 17-20, where its first letter is read as the altLoc.
    """
    lines, serial = [], 1
    for i, name in enumerate(names, start=1):
        atoms = _BACKBONE + (_SIDECHAIN if sidechain and i == 2 else [])
        record = "HETATM" if hetatm_idx == i else "ATOM  "
        for atom, x, y, z in atoms:
            lines.append(
                f"{record}{serial:5d} {atom:<4s} {name:<4s}A{i:4d}    "
                f"{x + 3.8 * i:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2s}"
            )
            serial += 1
    path = tmp_path / "res.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


class TestProtonationVariantsAreDecodablePositions:
    """Tier A: same heavy atoms as the parent, so ProDy registration is a pure win."""

    @pytest.mark.parametrize("resname", sorted(PROTONATION_VARIANT_3TO1))
    def test_prody_knows_it(self, resname: str) -> None:
        assert resname in flagDefinition("protein"), (
            f"{resname} is not in ProDy's amino-acid set, so it contributes no CA atom: the "
            "position is dropped and its atoms become ligand context."
        )

    @pytest.mark.parametrize("resname", sorted(PROTONATION_VARIANT_3TO1))
    def test_it_survives_as_exactly_one_position(self, tmp_path, resname: str) -> None:
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"]))[0]
        assert len(parsed["S"]) == 3, f"{resname} cost a position: {len(parsed['S'])} != 3"

    @pytest.mark.parametrize("resname", sorted(PROTONATION_VARIANT_3TO1))
    def test_its_side_chain_does_not_become_ligand_context(self, tmp_path, resname: str) -> None:
        """Equality, not <=, and with a real side chain — see the module docstring."""
        alias = np.asarray(
            parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"], sidechain=True),
                      parse_all_atoms=True)[0]["Y_t"]
        )
        parent = np.asarray(
            parse_PDB(_pdb(tmp_path, ["ALA", _1TO3[PROTONATION_VARIANT_3TO1[resname]], "ALA"],
                           sidechain=True), parse_all_atoms=True)[0]["Y_t"]
        )
        assert alias.shape[0] == parent.shape[0], (
            f"{resname} contributed {alias.shape[0] - parent.shape[0]} ligand-context atoms "
            f"its parent does not: it was classified as 'not protein and not water' and "
            "handed to the model as part of the substrate."
        )


class TestModifiedResiduesAreNotRegistered:
    """Tier B: the guard that keeps a cofactor visible.

    Naming a modified residue ``protein`` does not gain its extra atoms a decode slot — they
    are not among the 37 — and loses them as ligand context, so the model sees neither. This
    test fails if anyone moves a tier-B code into tier A.
    """

    # ↓ Derived from ProDy, not hard-coded: the names ProDy already knows (MSE, PTR, SEP, ...)
    #   are `protein` regardless of anything here, and listing them by hand would turn a
    #   ProDy upgrade into a spurious failure — and would need editing every time the table
    #   grows, which is exactly when this guard matters most.
    @pytest.mark.parametrize(
        "resname",
        sorted(n for n in MODIFIED_RESIDUE_3TO1 if n not in _PRODY_NATIVE),
    )
    def test_not_newly_registered_with_prody(self, resname: str) -> None:
        assert resname not in flagDefinition("protein"), (
            f"{resname} is a structurally modified residue and must NOT be registered: doing "
            "so makes its modification invisible to the model rather than substrate context. "
            "Measured for LLP: 16 ligand-context atoms unregistered, 1 registered."
        )

    def test_a_cofactor_bearing_residue_keeps_its_atoms(self, tmp_path) -> None:
        """LLP is on Xyme's live variant panel, so this is the case that matters."""
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", "LLP", "ALA"], sidechain=True),
                           parse_all_atoms=True)[0]
        assert np.asarray(parsed["Y_t"]).shape[0] > 1, (
            "LLP's cofactor atoms reached the model as neither a position nor ligand context"
        )


class TestIdentityIsTheParentNotUnknown:
    @pytest.mark.parametrize("resname,expected", sorted(NONCANONICAL_3TO1.items()))
    def test_alias_decodes_as_its_parent(self, tmp_path, resname: str, expected: str) -> None:
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", resname, "ALA"]))[0]
        S = np.asarray(parsed["S"])
        if resname not in flagDefinition("protein"):
            pytest.skip(f"{resname} is tier B and unregistered by design; identity unreachable")
        reference = np.asarray(
            parse_PDB(_pdb(tmp_path, ["ALA", _1TO3[expected], "ALA"]))[0]["S"]
        )
        assert S[1] != UNKNOWN_TOKEN, f"{resname} decoded as X, not {expected}"
        assert S[1] == reference[1], (
            f"{resname} decoded as token {S[1]}; its parent {expected} is {reference[1]}"
        )


class TestFreeAminoAcidLigandsAreNotStolen:
    """The inverse mistake: a free amino acid as SUBSTRATE must stay ligand context.

    ProDy's ``protein`` flag tests the residue name with no ATOM/HETATM discrimination, so
    registering a name that is also a plausible free ligand could turn a substrate into a
    spurious protein position — which matters most for the campaigns where free amino acids
    ARE the substrate.
    """

    @pytest.mark.parametrize("resname", ["ORN", "NVA", "NLE", "ABA", "PCA"])
    def test_free_amino_acid_ligand_stays_a_ligand(self, tmp_path, resname: str) -> None:
        parsed = parse_PDB(_pdb(tmp_path, ["ALA", "ALA", resname], hetatm_idx=3),
                           parse_all_atoms=True)[0]
        assert len(parsed["S"]) == 2, (
            f"a free {resname} HETATM became a protein position; it is substrate, not chain"
        )


class TestTheLiveVariantPanel:
    """The panel xyme-tools-rosetta's own tests bundle together, all six of it."""

    @pytest.mark.parametrize("resname", ["HIP", "ASH", "GLH", "LYN", "SED", "LLP"])
    def test_panel_member_has_a_canonical_identity(self, resname: str) -> None:
        assert resname in NONCANONICAL_3TO1, f"{resname} is on the live panel but unmapped"


class TestTheTableAgreesWithXymeToolsDatum:
    """``ALT_RESIDUE_DETAILS`` is the ecosystem's alias authority; these must not drift.

    Skips where xyme-tools-datum is absent, which includes CI unless it is installed there —
    stated plainly rather than presented as an unconditional guarantee.
    """

    def test_no_alias_contradicts_the_authority(self) -> None:
        datum = pytest.importorskip("xyme_tools_datum")
        authority = {
            code: detail.get("one_letter") if isinstance(detail, dict)
            else getattr(detail, "one_letter", None)
            for code, detail in datum.ALT_RESIDUE_DETAILS.items()
        }
        disagreements = {
            code: (ours, authority[code])
            for code, ours in NONCANONICAL_3TO1.items()
            if code in authority and authority[code] is not None and authority[code] != ours
        }
        assert not disagreements, (
            f"parent assignments disagree with ALT_RESIDUE_DETAILS: {disagreements}"
        )


class TestRegistrationSurvivesShadowedProDySettings:
    """The failure mode that is invisible at runtime and machine-specific.

    ``flags.updateDefinitions`` reads ``SETTINGS[NONSTANDARD_KEY]`` in preference to the
    module-level table, falling back to it only on ``KeyError``. Any earlier
    ``addNonstdAminoacid`` / ``delNonstdAminoacid`` / ``resetDefinitions`` call writes that key
    to the user's ``~/.prodyrc`` permanently — so on a machine where a scientist has ever
    customised ProDy, mutating the module table alone registers nothing, and every tier-A
    residue silently reverts to being dropped into ligand context.

    Everything here is in memory: nothing calls ``SETTINGS.save()``, and the fixture restores
    ProDy's definitions afterwards, because poisoning a developer's real settings file is the
    same bug seen from the other side.
    """

    @pytest.fixture
    def shadowed_settings(self):
        """A SETTINGS entry that does NOT contain our names, as a fresh machine's would not.

        Seeded from ProDy's table minus tier A: seeding from it wholesale would already carry
        our aliases (the module registers them at import), so the fixture would not reproduce
        the broken machine at all and the test would pass vacuously.

        Managed without monkeypatch because the teardown must run BEFORE the SETTINGS entry is
        removed — ``updateDefinitions`` reads it, so restoring in the wrong order leaves ProDy
        holding a half-built table for the rest of the session.
        """
        from prody.atomic import flags

        key = flags.NONSTANDARD_KEY
        had_key = key in flags.SETTINGS._settings
        previous = flags.SETTINGS._settings.get(key)
        original_module_table = {k: set(v) for k, v in flags.NONSTANDARD.items()}
        seeded = {
            k: set(v)
            for k, v in flags.NONSTANDARD.items()
            if k not in PROTONATION_VARIANT_3TO1
        }
        flags.SETTINGS._settings[key] = seeded
        flags.updateDefinitions()
        try:
            yield seeded
        finally:
            if had_key:
                flags.SETTINGS._settings[key] = previous
            else:
                flags.SETTINGS._settings.pop(key, None)
            flags.NONSTANDARD.clear()
            flags.NONSTANDARD.update(original_module_table)
            flags.updateDefinitions()

    def test_registration_writes_where_prody_will_actually_read(self, shadowed_settings) -> None:
        from ligandmpnn.data import _register_protonation_variants_with_prody

        assert "SED" not in shadowed_settings, "fixture failed to reproduce a clean machine"
        assert "SED" not in flagDefinition("protein"), (
            "the shadowed SETTINGS should have hidden our registration; if this fails the "
            "fixture is not reproducing the bug and the test below proves nothing"
        )
        _register_protonation_variants_with_prody()
        assert "SED" in flagDefinition("protein"), (
            "registration did not reach ProDy through the SETTINGS branch, so tier-A residues "
            "would be silently dropped on any machine whose ~/.prodyrc carries "
            "flags_nonstandard"
        )

    def test_it_does_not_persist_anything_to_disk(self, shadowed_settings, monkeypatch) -> None:
        """A parser import must not write to the user's ProDy settings file."""
        from prody.atomic import flags
        from ligandmpnn.data import _register_protonation_variants_with_prody

        saves: list[int] = []
        monkeypatch.setattr(flags.SETTINGS, "save", lambda *a, **k: saves.append(1))
        _register_protonation_variants_with_prody()
        assert saves == [], "registration called SETTINGS.save(), persisting to ~/.prodyrc"
