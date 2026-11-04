# Changelog

## 1.5.0

### Added

`ProteinMPNN.unconditional_probs` and `score.py --unconditional_probs_only 1`: the
per-position amino-acid distribution given the backbone alone, from one deterministic
decoder pass. No residue identity reaches the decoder and no decoding order is involved.

It is numerically identical to `single_aa_score(use_sequence=True)`, which reaches the same
quantity in L decoder passes. In that mode the scored position is decoded first, so it sees
no neighbour residue; despite the flag's name, `use_sequence=False` decodes it last, with
full context. The equivalence is pinned by a unit test.

The `.pt` written by `score.py` gains a `mode` key (`autoregressive`, `single_aa` or
`unconditional`). In the unconditional mode `decoding_order` and `std_of_probs` are absent,
because a single deterministic pass has neither, and `post_processing` states that the
output is the raw 21-way softmax (column 20 is `X`) with no bias, omit list or temperature.

**The mode refuses what it cannot honour.** `score.py` exits on `--autoregressive_score`,
`--fixed_residues*`, `--redesigned_residues*`, `--symmetry_residues`, `--homo_oligomer` and
`--chains_to_design`, which the pass would otherwise ignore without a word, and forces
`batch_size` and `number_of_batches` to 1. `unconditional_probs` raises for `ligand_mpnn`
with side-chain context when any residue is fixed: the encoder ingests those native side
chains, which is sequence information.

### Changed

`score.py` builds its parser in `build_argparser()` and reads `unconditional_probs_only`
with `getattr`, so a hand-built `argparse.Namespace` from before this release keeps working.

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

### Fixed

The test suite now passes. Both long-standing failures were in the harness, not the model.

`xyme-lake` moves from `>=0.16.0` to `>=0.23.0`. 0.23.0 is the first release carrying
`xyme_lake/sso.py`, the profile-less path that discovers `*TeamAccess` roles from the
cached SSO token and assumes `dev-DataAccess-<team>` from there. On 0.16.0 the assume-role
is attempted from whatever ambient profile is configured, which fails with `AccessDenied`
on every current Xyme SSO session — so `params.download_models` could not fetch a single
checkpoint. The old floor resolved to exactly 0.16.0, so nothing warned.

`test_protein_mpnn_output_unchanged` compared the whole FASTA header against its fixture,
including `model_path=`, which is the absolute path of the machine that minted it
(`/Users/eclay/...`). The test could only ever pass there. The comparison now normalises
that one field; every other header value, and both sequences, are still asserted exactly.

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
