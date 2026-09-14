# Native OpenFold3 reevaluation

`openfold3` is an optional external evaluator. AF3 remains the default. The
native adapter is tested against OpenFold3 **0.4.3**, Git revision
`0bb17be5199846e806b6347b6e17c6249c88ff1b`, with its compatible Preview-2
checkpoint. Install it in a separate environment; no OpenFold3 or PyTorch
packages are required in the ODIN design environment.

## Installation and configuration

Copy [`settings_reevaluation/openfold3.example.json`](../settings_reevaluation/openfold3.example.json) and configure the executable,
its environment Python, a staged checkpoint, runtime cache, and target-MSA
cache. Paths are relative to the configuration file unless absolute. The
adapter records the installed version/source and checkpoint checksum. Do not
mix software or checkpoints under an existing named evaluation.

Validate the configured installation before preprocessing:

```bash
python validate_install.py --openfold3-config /path/to/openfold3.local.json
```

## Prepare targets and reevaluate

Run commands from the repository root:

```bash
python odin_multi.py preprocess-openfold3 \
  --settings /path/to/target.json \
  --evaluator-config /path/to/openfold3.local.json
python odin_multi.py evaluate --run-dir outputs/my_run \
  --selection last --evaluation-name native \
  --evaluator openfold3 \
  --evaluator-config /path/to/openfold3.local.json
python odin_multi.py summarize --run-dir outputs/my_run
```

Preprocessing sends only fixed target sequences to the configured ColabFold
service. Target alignments are checksummed and reused. The designed sequence
remains query-only; templates and MSA-server access are explicitly disabled
during inference. Missing or changed cache files fail before prediction. The
adapter currently accepts canonical protein chains, validating the output
sequence and atom/token dimensions before computing metrics. Native one-based
sample IDs become zero-based IDs in ODIN tables. Binder iPTM is the mean of the
available chain-pair scores across target chains; the native
whole-complex value remains `global_i_ptm`. Missing binder-pair scores fail
rather than using the whole-complex value.

This follows AF3's cross-chain aggregation definition. Native 0.4.3 emits one
score per unordered chain pair. See [AF3 integration compatibility](../OPENFOLD3_COMPATIBILITY.md)
for the matched lifecycle, regression evidence, and explicit MSA/template
protocol differences.

## Parallel workers and resume

Use `--shard-index` and `--num-shards` to spread independent designs across
workers, as with AF3. Each worker must receive its own GPU allocation. The native
runner is fixed to one device and one node; increasing the number of workers
does not pool GPU memory for a single trajectory. Completed jobs are reused only
when their immutable inputs and saved artifact checksums match. Failed attempts
and their logs are retained under the job directory.

`runner_settings` accepts explicit `model_update`, `data_module_args`, and
`dataset_config_kwargs` sections. Paths, seeds, output format, device count, and
MSA/template policy are owned by the adapter. Optional interface scoring follows
the AF3 configuration fields and requires PyRosetta in the ODIN environment.

Results are written to `03_evaluations/openfold3/NAME/`. The metrics table
contains one row per selected design, context, seed, and diffusion sample.
Completion requires every configured sample and its structure and confidence
artifacts. Summary generation uses the same command as AF3.

Set `ODIN_TIMING_LOG` to a writable JSONL path to record native subprocess
timing events. Native prediction timing files remain alongside model outputs.
