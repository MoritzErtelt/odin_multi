"""Native OpenFold3 subprocess backend; no AWS or GPU imports in this module.

Supported input is canonical protein chains. Native outputs use one-based sample
IDs; the ODIN result contract uses zero-based IDs, matching its other evaluators.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from odin_multi import (
    _csv_text,
    _slug,
    atomic_write_json,
    atomic_write_text,
    file_lock,
    load_json,
    load_run,
)
from run_layout import layout_for_run

from .af3 import _selection, _sequence, _settings_target, _target_chains
from .interface import INTERFACE_EXTRA_FIELDS, INTERFACE_FIELDS, score_structure
from .ipsae import compute_ipsae_min

BACKEND = "openfold3"
SCHEMA = 1
METRICS = [
    "plddt",
    "binder_plddt",
    "ptm",
    "i_ptm",
    "global_i_ptm",
    "i_pae",
    "min_i_pae",
    "ipsae_min",
    "ranking_score",
]
FIELDS = [
    "design_index",
    "design_id",
    "selection",
    "selected_iteration",
    "selected_stage",
    "sequence",
    "context",
    "role",
    "seed",
    "sample",
    *METRICS,
    *INTERFACE_FIELDS,
    *INTERFACE_EXTRA_FIELDS,
    "structure",
]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty filesystem path")
    p = Path(value).expanduser()
    # Resolving a venv Python symlink would bypass its pyvenv.cfg.
    return Path(os.path.abspath(base / p if not p.is_absolute() else p))


def normalize_config(path: Path, *, require_model: bool = True) -> dict[str, Any]:
    raw = load_json(path)
    allowed = {
        "executable",
        "python",
        "checkpoint",
        "cache_dir",
        "target_cache_dir",
        "seeds",
        "num_diffusion_samples",
        "timeout_seconds",
        "msa_server_url",
        "runner_settings",
        "interface_metrics",
        "interface_relax",
    }
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Unknown native OpenFold3 configuration fields")
    base = path.resolve().parent
    executable = raw.get("executable", "run_openfold")
    if os.sep not in executable:
        executable = shutil.which(executable)
    executable = _path(executable, base)
    python = _path(raw.get("python", str(executable.parent / "python")), base)
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not python.is_file()
    ):
        raise FileNotFoundError(
            "OpenFold3 executable and environment Python must exist"
        )
    seeds = raw.get("seeds", [1])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    config = {
        **raw,
        "executable": str(executable),
        "python": str(python),
        "seeds": seeds,
    }
    for key, default in (("num_diffusion_samples", 5), ("timeout_seconds", 21600)):
        value = raw.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        config[key] = value
    for key in ("checkpoint", "cache_dir", "target_cache_dir"):
        config[key] = str(_path(raw.get(key), base))
    if require_model and not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError("OpenFold3 checkpoint must be staged before inference")
    config["msa_server_url"] = raw.get("msa_server_url", "https://api.colabfold.com")
    if not isinstance(config["msa_server_url"], str) or not config[
        "msa_server_url"
    ].startswith(("https://", "http://")):
        raise ValueError("msa_server_url must be HTTP(S)")
    for key, default in (("interface_metrics", False), ("interface_relax", True)):
        config[key] = raw.get(key, default)
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    settings = raw.get("runner_settings", {})
    # Workflow-owned paths, sample identity, device count and input policy may
    # not be changed via runner settings. Model settings remain explicit.
    if not isinstance(settings, dict) or set(settings) - {
        "model_update",
        "data_module_args",
        "dataset_config_kwargs",
    }:
        raise ValueError(
            "runner_settings supports model_update, data_module_args and dataset_config_kwargs only"
        )
    config["runner_settings"] = settings
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m,json; d=m.distribution('openfold3'); print(json.dumps({'version':d.version,'source':d.read_text('direct_url.json')}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ),
    )
    config["software"] = json.loads(probe.stdout)
    config["checkpoint_sha256"] = (
        file_sha256(Path(config["checkpoint"])) if require_model else None
    )
    return config


def cache_key(chains: list[dict], config: dict) -> str:
    return digest(
        {
            "schema": SCHEMA,
            "chains": chains,
            "server": config["msa_server_url"],
            "software": config["software"],
            "templates": False,
        }
    )


def _run(config: dict, arguments: list[str], work: Path, env: dict[str, str]) -> None:
    from runtime_events import emit

    work.mkdir(parents=True, exist_ok=True)
    emit("native_command_started", command=arguments[0], work=str(work))
    try:
        with (
            (work / "stdout.log").open("w") as out,
            (work / "stderr.log").open("w") as err,
        ):
            subprocess.run(
                [config["executable"], *arguments],
                check=True,
                timeout=config["timeout_seconds"],
                stdout=out,
                stderr=err,
                env=env,
            )
    finally:
        emit("native_command_finished", command=arguments[0], work=str(work))


def _cached(chains: list[dict], config: dict) -> tuple[str, list[dict]]:
    key = cache_key(chains, config)
    root = Path(config["target_cache_dir"]) / key
    meta = load_json(root / "metadata.json")
    if meta.get("key") != key or meta.get("status") != "complete":
        raise ValueError("Incomplete or incompatible OpenFold3 target cache")
    for relative, checksum in meta["files"].items():
        p = (root / relative).resolve()
        if (
            not p.is_relative_to(root.resolve())
            or not p.is_file()
            or file_sha256(p) != checksum
        ):
            raise ValueError("Missing or changed target MSA cache artifact")
    features = copy.deepcopy(meta["chains"])
    if [(c["chain_ids"], c["sequence"]) for c in features] != [
        ([c["chain_id"]], c["sequence"]) for c in chains
    ]:
        raise ValueError("Cached chain identity mismatch")
    referenced = {
        p
        for chain in features
        for field in ("main_msa_file_paths", "paired_msa_file_paths")
        for p in chain.get(field, [])
    }
    if set(meta["files"]) != referenced or any(
        not chain.get("main_msa_file_paths") for chain in features
    ):
        raise ValueError("Target MSA cache artifact coverage is incomplete")
    for chain in features:
        for field in ("main_msa_file_paths", "paired_msa_file_paths"):
            chain[field] = [str(root / p) for p in chain.get(field, [])]
    return key, features


def preprocess_targets(
    settings_paths: list[Path], *, config_path: Path
) -> dict[str, int]:
    config = normalize_config(config_path, require_model=False)
    counts = {"completed": 0, "skipped": 0}
    for settings in settings_paths:
        _, chains = _settings_target(settings)
        key = cache_key(chains, config)
        root = Path(config["target_cache_dir"]) / key
        with file_lock(root.parent / ".locks" / f"{key}.lock"):
            if (root / "metadata.json").exists():
                _cached(chains, config)
                counts["skipped"] += 1
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
            query = {
                "queries": {
                    "target": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": [c["chain_id"]],
                                "sequence": c["sequence"],
                            }
                            for c in chains
                        ]
                    }
                }
            }
            atomic_write_json(attempt / "query.json", query)
            atomic_write_json(
                attempt / "msa.yml",
                {
                    "server_url": config["msa_server_url"],
                    "msa_file_format": "a3m",
                    "cleanup_msa_dir": False,
                },
            )
            _run(
                config,
                [
                    "align-msa-server",
                    "--query-json",
                    str(attempt / "query.json"),
                    "--output-dir",
                    str(attempt / "alignments"),
                    "--msa-computation-settings-yaml",
                    str(attempt / "msa.yml"),
                ],
                attempt,
                dict(os.environ),
            )
            features = load_json(attempt / "alignments/query_msa.json")["queries"][
                "target"
            ]["chains"]
            saved, files = [], {}
            for original, chain in zip(chains, features, strict=True):
                if chain["sequence"] != original["sequence"] or chain["chain_ids"] != [
                    original["chain_id"]
                ]:
                    raise ValueError("MSA preprocessing changed chain identity")
                result = {
                    "molecule_type": "protein",
                    "chain_ids": chain["chain_ids"],
                    "sequence": chain["sequence"],
                }
                for field in ("main_msa_file_paths", "paired_msa_file_paths"):
                    paths = chain.get(field) or []
                    if field == "main_msa_file_paths" and not paths:
                        raise ValueError(
                            "Target preprocessing returned no main alignment"
                        )
                    result[field] = []
                    for value in paths:
                        source = Path(value).resolve()
                        if not source.is_file() or not source.is_relative_to(
                            root.resolve()
                        ):
                            raise ValueError(
                                "MSA path must be a file inside the cache attempt"
                            )
                        text = source.read_text()
                        query_seq = "".join(text.splitlines()[1:]).split(">")[0]
                        query_seq = re.sub(r"[a-z.\-]", "", query_seq)
                        if query_seq != original["sequence"]:
                            raise ValueError("MSA query sequence mismatch")
                        relative = str(source.relative_to(root.resolve()))
                        files[relative] = file_sha256(source)
                        result[field].append(relative)
                saved.append(result)
            atomic_write_json(
                root / "metadata.json",
                {"status": "complete", "key": key, "chains": saved, "files": files},
            )
            _cached(chains, config)
            counts["completed"] += 1
    return counts


def runner_settings(config: dict, output: Path) -> dict:
    settings = copy.deepcopy(config["runner_settings"])
    settings.update(
        {
            "experiment_settings": {
                "seeds": config["seeds"],
                "use_msa_server": False,
                "use_templates": False,
                "output_dir": str(output),
            },
            "pl_trainer_args": {"devices": 1, "num_nodes": 1, "accelerator": "gpu"},
            # Native 0.4.3 passes this keyword explicitly from the CLI. Including
            # it in YAML as well raises TypeError before the runner is created.
            "cache_path": config["cache_dir"],
            "output_writer_settings": {
                "structure_format": "cif",
                "write_full_confidence_scores": True,
                "full_confidence_output_format": "json",
                "write_latent_outputs": False,
            },
        }
    )
    return settings


def _biopython_cif_text(structure: Path) -> str:
    """Supply Biopython's required occupancy for native single-conformer models.

    OpenFold3 0.4.3 omits this optional mmCIF column. Unit occupancy is a
    parser convention for the predicted conformation, not a measured quantity.
    The native artifact, coordinates, atom order, and confidence stay unchanged.
    """
    from Bio.PDB import MMCIFIO
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    text = structure.read_text()
    data = MMCIF2Dict(io.StringIO(text))
    if "_atom_site.occupancy" in data:
        return text
    if any(value not in {".", "?"} for value in data["_atom_site.label_alt_id"]):
        raise ValueError("Cannot default occupancy for alternate conformations")
    data["_atom_site.occupancy"] = ["1.0"] * len(data["_atom_site.id"])
    writer, output = MMCIFIO(), io.StringIO()
    writer.set_dict(data)
    writer.save(output)
    return output.getvalue()


def read_native_structure(structure: Path):
    """Read a native CIF without modifying the original prediction artifact."""
    from Bio.PDB import MMCIFParser

    return MMCIFParser(QUIET=True).get_structure(
        "prediction", io.StringIO(_biopython_cif_text(structure))
    )


def compute_metrics(
    summary: dict, confidence: dict, structure: Path, chains: list[dict], binder_id: str
) -> dict:
    from Bio.SeqUtils import seq1

    model = read_native_structure(structure)[0]
    expected = {c["chain_ids"][0]: c["sequence"] for c in chains}
    if {c.id for c in model} != set(expected):
        raise ValueError("Predicted chain IDs differ from query")
    token_ids, atom_ids = [], []
    for chain in model:
        residues = list(chain.get_residues())
        if any(r.id[0] != " " or "CA" not in r or r.is_disordered() for r in residues):
            raise ValueError("Expected canonical protein residues with unique CA atoms")
        if "".join(seq1(r.resname) for r in residues) != expected[chain.id]:
            raise ValueError("Predicted chain sequence differs from query")
        for residue in residues:
            token_ids.append(chain.id)
            atom_ids.extend([chain.id] * len(list(residue.get_atoms())))
    pae = np.asarray(confidence["pae"], dtype=float)
    plddt = np.asarray(confidence["plddt"], dtype=float)
    if pae.shape != (len(token_ids), len(token_ids)) or plddt.shape != (len(atom_ids),):
        raise ValueError(
            "Confidence dimensions do not match canonical protein tokens/atoms"
        )
    if (
        not np.isfinite(pae).all()
        or (pae < 0).any()
        or not np.isfinite(plddt).all()
        or (plddt < 0).any()
        or (plddt > 100).any()
    ):
        raise ValueError("Invalid confidence values")
    # Native OpenFold3 pLDDT is explicitly on the 0..100 scale.
    binder, target = (
        np.asarray(token_ids) == binder_id,
        np.asarray(token_ids) != binder_id,
    )
    interfaces = np.concatenate(
        (pae[np.ix_(binder, target)].ravel(), pae[np.ix_(target, binder)].ravel())
    )
    pairs = summary["chain_pair_iptm"]
    pair_values = []
    for target_id in expected.keys() - {binder_id}:
        values = [
            pairs[k]
            for k in (f"({binder_id}, {target_id})", f"({target_id}, {binder_id})")
            if k in pairs
        ]
        if not values:
            raise ValueError("Missing binder-to-target chain-pair iPTM")
        pair_values.append(float(np.mean(values)))
    result = {
        "plddt": float(plddt.mean() / 100),
        "binder_plddt": float(plddt[np.asarray(atom_ids) == binder_id].mean() / 100),
        "ptm": float(summary["ptm"]),
        "global_i_ptm": float(summary["iptm"]),
        "i_ptm": float(np.mean(pair_values)),
        "i_pae": float(interfaces.mean()),
        "min_i_pae": float(interfaces.min()),
        "ipsae_min": compute_ipsae_min(pae, binder, target),
        "ranking_score": float(summary["sample_ranking_score"]),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite aggregate confidence")
    return result


def collect_output(
    output: Path, config: dict, chains: list[dict], binder: str
) -> list[dict]:
    samples = []
    for path in sorted(output.rglob("*_confidences_aggregated.json")):
        match = re.search(
            r"_seed_(\d+)_sample_(\d+)_confidences_aggregated.json$", path.name
        )
        if match is None:
            raise ValueError("Unrecognized OpenFold3 sample filename")
        seed, sample = int(match[1]), int(match[2]) - 1
        stem = path.name.removesuffix("_confidences_aggregated.json")
        full, structure = (
            path.with_name(stem + "_confidences.json"),
            path.with_name(stem + "_model.cif"),
        )
        metrics = compute_metrics(
            load_json(path), load_json(full), structure, chains, binder
        )
        if config["interface_metrics"]:
            with tempfile.TemporaryDirectory(
                prefix="odin-openfold3-interface-"
            ) as temp:
                compatible = Path(temp) / (stem + "_model.cif")
                compatible.write_text(_biopython_cif_text(structure))
                metrics.update(
                    score_structure(
                        compatible,
                        binder,
                        [
                            c["chain_ids"][0]
                            for c in chains
                            if c["chain_ids"] != [binder]
                        ],
                        relax=config["interface_relax"],
                        workdir=path.parent / (stem + "_interface"),
                    )
                )
        samples.append(
            {
                "seed": seed,
                "sample": sample,
                "metrics": metrics,
                "structure": str(structure),
                "confidences": str(full),
                "summary_confidences": str(path),
            }
        )
    observed = [(s["seed"], s["sample"]) for s in samples]
    expected = {
        (seed, sample)
        for seed in config["seeds"]
        for sample in range(config["num_diffusion_samples"])
    }
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            f"Incomplete or duplicate OpenFold3 samples: expected {sorted(expected)}, found {observed}"
        )
    return samples


def complete(path: Path, run_dir: Path, identity: str, config: dict) -> bool:
    if not path.is_file():
        return False
    result = load_json(path)
    if result.get("identity") != identity or result.get("status") != "complete":
        return False
    expected = {
        (seed, sample)
        for seed in config["seeds"]
        for sample in range(config["num_diffusion_samples"])
    }
    observed = [
        (item.get("seed"), item.get("sample")) for item in result.get("samples", [])
    ]
    if not expected or len(observed) != len(expected) or set(observed) != expected:
        return False
    required = [
        sample.get(field)
        for sample in result["samples"]
        for field in ("structure", "confidences", "summary_confidences")
    ]
    if (
        any(not isinstance(value, str) or not value for value in required)
        or len(set(required)) != 3 * len(expected)
        or set(required) != set(result.get("artifacts", {}))
    ):
        return False
    for relative, checksum in result.get("artifacts", {}).items():
        p = (run_dir / relative).resolve()
        if (
            not p.is_relative_to(run_dir.resolve())
            or not p.is_file()
            or file_sha256(p) != checksum
        ):
            return False
    return bool(result.get("samples")) and bool(result.get("artifacts"))


def run_evaluation(
    run_dir: Path,
    selection_name: str,
    evaluation_name: str,
    *,
    config_path: Path,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, int]:
    if (
        num_shards < 1
        or not 0 <= shard_index < num_shards
        or _slug(evaluation_name) != evaluation_name
    ):
        raise ValueError("Invalid shard or evaluation name")
    run_dir, manifest = load_run(run_dir)
    selection = _selection(run_dir, selection_name)
    config = normalize_config(config_path)
    cached = {
        c["name"]: _cached(_target_chains(run_dir, c), config)
        for c in manifest["contexts"]
    }
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir(BACKEND, evaluation_name)
    metadata = {
        "schema_version": SCHEMA,
        "evaluator": BACKEND,
        "evaluation_name": evaluation_name,
        "selection_name": selection_name,
        "selection_sha256": digest(selection),
        "config": config,
        "target_cache_keys": {k: v[0] for k, v in cached.items()},
    }
    with file_lock(layout.locks / BACKEND / evaluation_name / "configure.lock"):
        p = root / "evaluation.json"
        if p.exists() and load_json(p) != metadata:
            raise ValueError(
                "Named evaluation inputs changed; choose a new evaluation name"
            )
        atomic_write_json(p, metadata)
    env = dict(os.environ)
    counts = {"completed": 0, "skipped": 0, "busy": 0, "failed": 0}
    for row in selection["rows"]:
        if (
            row["selection_status"] != "selected"
            or int(row["design_index"]) % num_shards != shard_index
        ):
            continue
        for context in manifest["contexts"]:
            _key, target = cached[context["name"]]
            job = (
                layout.evaluation_jobs(BACKEND, evaluation_name)
                / f"t{int(row['design_index']):05d}"
                / _slug(context["name"])
            )
            identity = digest({"metadata": metadata, "row": row, "context": context})
            with file_lock(
                layout.locks
                / BACKEND
                / evaluation_name
                / f"{job.parent.name}_{job.name}.lock",
                blocking=False,
            ) as acquired:
                if not acquired:
                    counts["busy"] += 1
                    continue
                if complete(job / "result.json", run_dir, identity, config):
                    counts["skipped"] += 1
                    continue
                job.mkdir(parents=True, exist_ok=True)
                attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=job))
                try:
                    sequence = _sequence(row["sequence"])
                    binder = next(
                        c
                        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                        if c not in {t["chain_ids"][0] for t in target}
                    )
                    chains = copy.deepcopy(target) + [
                        {
                            "molecule_type": "protein",
                            "chain_ids": [binder],
                            "sequence": sequence,
                        }
                    ]
                    atomic_write_json(
                        attempt / "query.json",
                        {"queries": {"prediction": {"chains": chains}}},
                    )
                    output = attempt / "outputs"
                    atomic_write_json(
                        attempt / "runner.yml", runner_settings(config, output)
                    )
                    # Isolate implicit user defaults; preserve caller GPU visibility.
                    env["OPENFOLD_CACHE"] = str(attempt / "runtime-cache")
                    _run(
                        config,
                        [
                            "predict",
                            "--query-json",
                            str(attempt / "query.json"),
                            "--runner-yaml",
                            str(attempt / "runner.yml"),
                            "--inference-ckpt-path",
                            config["checkpoint"],
                            "--num-diffusion-samples",
                            str(config["num_diffusion_samples"]),
                            "--use-msa-server",
                            "false",
                            "--use-templates",
                            "false",
                            "--output-dir",
                            str(output),
                        ],
                        attempt,
                        env,
                    )
                    samples = collect_output(output, config, chains, binder)
                    artifacts = {}
                    for sample in samples:
                        for field in (
                            "structure",
                            "confidences",
                            "summary_confidences",
                        ):
                            p = Path(sample[field])
                            sample[field] = str(p.relative_to(run_dir))
                            artifacts[sample[field]] = file_sha256(p)
                    atomic_write_json(
                        job / "result.json",
                        {
                            "status": "complete",
                            "identity": identity,
                            "seeds": config["seeds"],
                            "num_diffusion_samples": config["num_diffusion_samples"],
                            "design_index": row["design_index"],
                            "design_id": row["design_id"],
                            "selection_name": selection_name,
                            "selected_iteration": row["iteration"],
                            "selected_stage": row["stage"],
                            "sequence": sequence,
                            "context": context,
                            "binder_chain": binder,
                            "samples": samples,
                            "artifacts": artifacts,
                        },
                    )
                    counts["completed"] += 1
                except Exception as error:  # noqa: BLE001 - persist per-job failures before failing the worker
                    atomic_write_json(
                        job / "failure.json",
                        {
                            "design_index": row["design_index"],
                            "context": context["name"],
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    counts["failed"] += 1
    if counts["failed"]:
        raise RuntimeError(
            f"{counts['failed']} OpenFold3 jobs failed; inspect failure.json"
        )
    return counts


def collect_evaluation(run_dir: Path, evaluation_name: str) -> dict[str, int]:
    run_dir, _ = load_run(run_dir)
    layout = layout_for_run(run_dir)
    root = layout.evaluation_dir(BACKEND, evaluation_name)
    metadata = load_json(root / "evaluation.json")
    selection = _selection(run_dir, metadata["selection_name"])
    if digest(selection) != metadata["selection_sha256"]:
        raise ValueError("Selection changed since this evaluation was configured")
    rows = {
        int(r["design_index"]): r
        for r in selection["rows"]
        if r["selection_status"] == "selected"
    }
    manifest = load_run(run_dir)[1]
    contexts = {_slug(c["name"]): c for c in manifest["contexts"]}
    results, failures = [], []
    with file_lock(layout.locks / BACKEND / evaluation_name / "collect.lock"):
        for job in sorted(
            layout.evaluation_jobs(BACKEND, evaluation_name).glob("t*/*")
        ):
            p = job / "result.json"
            result = load_json(p) if p.exists() else {}
            row = rows.get(int(job.parent.name[1:]))
            context = contexts.get(job.name)
            identity = digest({"metadata": metadata, "row": row, "context": context})
            if (
                row is not None
                and context is not None
                and complete(p, run_dir, identity, metadata["config"])
            ):
                for sample in result["samples"]:
                    results.append(
                        {
                            "design_index": result["design_index"],
                            "design_id": result["design_id"],
                            "selection": result["selection_name"],
                            "selected_iteration": result["selected_iteration"],
                            "selected_stage": result["selected_stage"],
                            "sequence": result["sequence"],
                            "context": result["context"]["name"],
                            "role": result["context"]["role"],
                            "seed": sample["seed"],
                            "sample": sample["sample"],
                            "structure": sample["structure"],
                            **sample["metrics"],
                        }
                    )
            elif (job / "failure.json").exists():
                failures.append(load_json(job / "failure.json"))
            else:
                failures.append(
                    {
                        "design_index": int(job.parent.name[1:]),
                        "context": job.name,
                        "error": "Missing or invalid result artifacts",
                    }
                )
        atomic_write_text(
            layout.evaluation_metrics(BACKEND, evaluation_name),
            _csv_text(results, FIELDS),
        )
        atomic_write_text(
            layout.evaluation_failures(BACKEND, evaluation_name),
            _csv_text(failures, ["design_index", "context", "error"]),
        )
    return {"rows": len(results), "failed": len(failures)}
