# Changelog

## 1.4.0

### Added

A checkpoint now describes its own architecture, and `models.build_model_from_checkpoint`
is the supported way to load one.

`__main__` previously read `num_edges` and `atom_context_num` from the checkpoint but
hardcoded `node_features`, `edge_features`, `hidden_dim`, `num_encoder_layers` and
`num_decoder_layers` as literal `128`/`3`/`3`. Any checkpoint trained at a different
width was therefore unloadable, however compatible its weights. Those five are now read
from the checkpoint via `models.model_kwargs_from_checkpoint`.

**The new keys are optional.** All nineteen shipped checkpoints carry none of them and
load unchanged on the historical 128/3+3 defaults (`models.LEGACY_ARCHITECTURE`). Callers
outside this package that construct `ProteinMPNN` by hand are unaffected, but should
migrate: a dimension table maintained apart from the weights it describes can disagree
with them silently.

An optional per-residue condition head, for condition-aware checkpoints such as Baker
lab's TMB_MPNN. `ProteinMPNN(num_conditions=N)` adds a zero-initialised
`nn.Embedding(N, hidden_dim)` whose output seeds `h_V` in `encode`. It is built only when
`num_conditions` is non-zero, so `load_state_dict` stays strict in both directions: an
unconditioned checkpoint gains no unexpected key, and a conditioned one is not silently
loaded into a model that would discard its condition weights.

Because `h_V` starts at zero, the embedding **is** the initial node state rather than a
perturbation of it, so discarding it is not a no-op — a conditioned model would decode
from a state it never saw in training. Passing labels to a model without a condition head
raises instead of ignoring them. Labels are `0` unlabelled, `1` interface, `2` buried,
`3` non-TMBB; an unlabelled run falls back to `NON_TMBB_CONDITION_LABEL`, matching
upstream TMB_MPNN's own inference default.

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
