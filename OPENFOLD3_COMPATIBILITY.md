# Native OpenFold3 and the existing AF3 evaluator

The adapter follows the evaluator lifecycle in the repository's original AF3
implementation (`evaluators/af3.py`, initial release commit `e4567e4`). It adds
an alternative predictor; it does not claim to reproduce AF3 predictions or
its complete preprocessing protocol. AF3 remains the default.

| Concern | Existing AF3 evaluator | Native OpenFold3 adapter |
| --- | --- | --- |
| Selection | Existing selection table, selected sequence and iteration | Same selection table and identities |
| Context coverage | Each selected sequence against every configured context | Same |
| Execution | External Python/AF3 installation | External native executable and its Python environment |
| CPU preparation | AF3 local database pipeline; cache target alignments and templates | Native alignment command using the configured ColabFold service; cache target alignments |
| Designed sequence | Query-only, empty paired/unpaired MSAs, no templates | Query-only, no binder MSA or templates |
| Target templates | Retained when returned by AF3 preprocessing | Disabled in this implementation |
| GPU preprocessing | Data pipeline disabled; cached target features required | MSA service and templates disabled; cached target alignments required |
| Replicates | Explicit seeds; five diffusion samples by default | Same defaults, explicit native sample-count argument |
| Parallel workers | Design index modulo shard count; per-job locks | Same assignment and lock granularity |
| Results | Per-job results, failure records, sample collection, summaries | Same lifecycle, with native file parsing and stricter checksum checks |
| Interface metrics | Shared interface PAE and iPSAE calculations; optional interface scoring | Same calculations and optional scorer |
| Whole-complex iPTM | Preserved separately as `global_i_ptm` | Same |

## Binder iPTM

ODIN's AF3 evaluator reads the binder entry of `chain_iptm`. AlphaFold3 derives
that array from the per-chain mean of off-diagonal chain-pair iPTM scores,
averaging rows and columns. See the upstream
[confidence implementation](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidences.py)
(`get_iptm_xchain` and `pae_metrics`) and
[output definitions](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md).

Native OpenFold3 0.4.3 computes symmetric chain-pair iPTM and serializes one key
per unordered pair. Averaging the binder's scores against the target chains
therefore gives the same aggregation definition. The adapter also accepts both
directions when provided. It never substitutes the whole-complex score for a
missing binder-to-target pair. This matches the aggregation, not the numerical
predictions of different models.

`test_multichain_aggregation_matches_af3_cross_chain_definition` checks a
multichain fixture with unequal target lengths and strong target-target scores.
It covers both directional and native unordered-pair representations, and
compares shared confidence/PAE/iPSAE parsing against ODIN's AF3 helper.

Native 0.4.3 CIF files omit the optional occupancy column. The adapter supplies
unit occupancy only in the parser's view of a single predicted conformation,
because Biopython requires that column. Native files, atom order, coordinates,
and confidence arrays remain unchanged. Alternate conformations without
occupancy are rejected. Optional interface scoring uses the same compatible
representation; unit occupancy is not an experimental measurement.
