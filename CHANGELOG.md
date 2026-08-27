# Changelog

## 1.3.1

### Fixed

`parse_PDB` no longer discards non-canonical residues or decodes them as `X`.

Two independent silent failures, both measured on a seven-state enzyme ensemble of 7 x 181
residues:

- A residue name ProDy does not know is not `protein`, so it contributes no CA atom: the
  position **disappeared** from the decode, and its atoms — "not protein and not water" — were
  handed to the model as **ligand context**. `SED` took the ensemble from 1267 positions and 95
  ligand atoms to 1266 and 101.
- A name outside `restype_3to1`, which held the canonical 20 alone, decoded as `X`. Four
  catalytic `HIP156` reached the model as unknown rather than histidine.

Protonation and charge variants (`ASH GLH ARN LYN CYM CYX TYM SED`, and the histidine states in
both the Rosetta/AMBER and CHARMM spellings) are now registered with ProDy and decode as their
parent. Structurally modified residues decode as their parent where ProDy already knew the
name, and are deliberately **not** newly registered: naming one `protein` gains its extra atoms
no decode slot and loses them as ligand context, so they would vanish from the model's input
entirely. Measured for `LLP` (PLP-lysine): 16 ligand-context atoms unregistered, 1 registered.

`SEC` is excluded — its parent `U` is not in the 20-letter alphabet.

**Behaviour change.** Pipelines setting a protonation variant now present it to the model as its
parent amino acid instead of `X`, so designs will differ from previous runs and results either
side of this version are not like-for-like.
