# Native OpenFold3 reevaluation

`openfold3` is an optional external evaluator. AF3 remains the default. The
native adapter is tested against OpenFold3 **0.4.3**, Git revision
`0bb17be5199846e806b6347b6e17c6249c88ff1b`, with its compatible Preview-2
checkpoint. Install it in a separate environment; no OpenFold3 or PyTorch
packages are required in the ODIN design environment.

## Installation and configuration

OpenFold3 runs in a separate Python environment. Keep the existing ODIN
environment for design, selection, and summary generation.

### 1. Install native OpenFold3

The adapter was tested with Python 3.12 and OpenFold3 0.4.3 at commit
`0bb17be5199846e806b6347b6e17c6249c88ff1b`. Use this revision for reproducible
installation.

```bash
conda create -n odin-openfold3 python=3.12 pip -y
conda activate odin-openfold3

python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  "openfold3 @ git+https://github.com/aqlaboratory/openfold-3.git@0bb17be5199846e806b6347b6e17c6249c88ff1b" \
  "numpy<2" "pandas<3"
```

This example uses the CUDA 12.8 PyTorch build. The NVIDIA driver must support
that runtime. For other hardware or CUDA environments, follow the
[upstream installation instructions](https://openfold-3.readthedocs.io/en/latest/Installation.html),
while retaining the OpenFold3 revision above.

Check the installation:

```bash
python -m pip check
run_openfold predict --help
run_openfold align-msa-server --help

python -c \
  "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Run the CUDA check inside a GPU allocation. GPU memory requirements depend on
the total input length and inference settings.

### 2. Download compatible model parameters

Run the upstream setup command from the OpenFold3 environment:

```bash
setup_openfold
```

Select the Preview-2 checkpoint, `of3-p2-155k.pt`, and record its absolute path.
OpenFold3 0.4.3 requires compatible parameters; the current upstream default
checkpoint may target a newer release. See the
[parameter compatibility documentation](https://openfold-3.readthedocs.io/en/latest/parameters_reference.html).

Complete parameter downloads before starting ODIN evaluation. ODIN requires
an existing checkpoint file and records its SHA-256 checksum.

### 3. Configure ODIN

Record the executable paths before leaving the OpenFold3 environment:

```bash
command -v run_openfold
command -v python
```

Return to the ODIN environment and repository directory. Copy the
[example configuration](../settings_reevaluation/openfold3.example.json):

```bash
cp settings_reevaluation/openfold3.example.json \
   settings_reevaluation/openfold3.local.json
```

Edit the local configuration:

```json
{
  "executable": "/absolute/path/to/odin-openfold3/bin/run_openfold",
  "python": "/absolute/path/to/odin-openfold3/bin/python",
  "checkpoint": "/absolute/path/to/models/of3-p2-155k.pt",
  "cache_dir": "/absolute/path/to/openfold3/cache",
  "target_cache_dir": "/absolute/path/to/odin_target_msas",
  "seeds": [1],
  "num_diffusion_samples": 5,
  "timeout_seconds": 21600,
  "msa_server_url": "https://api.colabfold.com",
  "runner_settings": {},
  "interface_metrics": false,
  "interface_relax": true
}
```

`executable` and `python` must refer to the same OpenFold3 environment.
`cache_dir` stores native runtime cache data; `target_cache_dir` stores reusable
fixed-target alignments. Both directories must be writable. Relative paths
resolve against the configuration file's directory.

Target preprocessing requires access to the configured ColabFold service and
sends fixed target sequences to that service. Evaluation uses the cached
alignments. Designed sequences remain query-only, and templates are disabled.

### 4. Validate the integration

Run from the ODIN environment:

```bash
python validate_install.py \
  --openfold3-config settings_reevaluation/openfold3.local.json
```

This runs ODIN's installation checks and checks the configured OpenFold3
executable, package metadata, and checkpoint file. It does not run an OpenFold3
prediction or establish that a particular input fits GPU memory.

Proceed with target preprocessing and reevaluation below.

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
