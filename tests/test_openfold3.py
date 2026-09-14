"""Native output contract fixtures, independent of an OpenFold3 installation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, PDBParser

from evaluators import openfold3 as of3
from odin_multi import build_parser


def config():
    return {
        "seeds": [7],
        "num_diffusion_samples": 2,
        "interface_metrics": False,
        "runner_settings": {},
        "checkpoint": "/models/pinned.pt",
        "cache_dir": "/cache",
        "msa_server_url": "https://example.test",
        "software": {"version": "test"},
    }


def sample(tmp_path, number, *, seed=7):
    stem = tmp_path / f"prediction_seed_{seed}_sample_{number}"
    pdb = tmp_path / "source.pdb"
    # Two one-residue chains, with deliberately different atom confidences.
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C  \nATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 90.00           C  \nEND\n"
    )
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("fixture", pdb))
    writer.save(str(stem) + "_model.cif")
    Path(str(stem) + "_confidences.json").write_text(
        json.dumps({"plddt": [80, 90], "pae": [[0, 4], [6, 0]]})
    )
    Path(str(stem) + "_confidences_aggregated.json").write_text(
        json.dumps(
            {
                "ptm": 0.8,
                "iptm": 0.95,
                "sample_ranking_score": 0.9,
                "chain_pair_iptm": {"(A, B)": 0.6, "(B, A)": 0.8},
            }
        )
    )


CHAINS = [{"chain_ids": ["A"], "sequence": "A"}, {"chain_ids": ["B"], "sequence": "G"}]


def test_native_samples_units_and_direction(tmp_path):
    sample(tmp_path, 1)
    sample(tmp_path, 2)
    values = of3.collect_output(tmp_path, config(), CHAINS, "B")
    assert [v["sample"] for v in values] == [0, 1]
    m = values[0]["metrics"]
    assert m["binder_plddt"] == 0.9
    assert m["i_ptm"] == pytest.approx(0.7)
    assert m["global_i_ptm"] == 0.95
    assert m["i_pae"] == 5


def test_native_cif_without_occupancy_preserves_artifact_and_scoring(
    tmp_path, monkeypatch
):
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    sample(tmp_path, 1)
    structure = tmp_path / "prediction_seed_7_sample_1_model.cif"
    data = MMCIF2Dict(structure)
    del data["_atom_site.occupancy"]  # Native OpenFold3 0.4.3 output contract.
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(structure))
    original = structure.read_bytes()
    observed = []

    def score(path, binder, targets, **kwargs):
        atoms = list(MMCIFParser(QUIET=True).get_structure("model", path).get_atoms())
        assert [a.occupancy for a in atoms] == [1.0, 1.0]
        np.testing.assert_array_equal([a.coord for a in atoms], [[0, 0, 0], [1, 0, 0]])
        observed.append((binder, targets))
        return {"interface_residues": "A1"}

    monkeypatch.setattr(of3, "score_structure", score)
    cfg = {
        **config(),
        "num_diffusion_samples": 1,
        "interface_metrics": True,
        "interface_relax": False,
    }
    result = of3.collect_output(tmp_path, cfg, CHAINS, "B")
    assert result[0]["metrics"]["i_pae"] == 5
    assert result[0]["structure"] == str(structure)
    assert observed == [("B", ["A"])]
    assert structure.read_bytes() == original
    data["_atom_site.label_alt_id"][0] = "A"
    writer.set_dict(data)
    writer.save(str(structure))
    with pytest.raises(ValueError, match="alternate conformations"):
        of3.read_native_structure(structure)


def test_multichain_aggregation_matches_af3_cross_chain_definition(tmp_path):
    from evaluators.af3 import compute_af3_metrics

    # Unequal target lengths distinguish per-chain from per-token weighting.
    pdb = tmp_path / "three_chains.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      3  CA  GLY B   2       2.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      4  CA  ALA C   1       3.000   0.000   0.000  1.00 80.00           C  \nEND\n"
    )
    structure = tmp_path / "three_chains.cif"
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("fixture", pdb))
    writer.save(str(structure))
    pairs = np.array([[0.9, 0.99, 0.2], [0.98, 0.9, 0.4], [0.6, 0.8, 0.9]])
    # AF3 get_iptm_xchain averages off-diagonal rows and columns per chain.
    expected = 0.5 * (pairs[2, :2].mean() + pairs[:2, 2].mean())
    summary = {
        "ptm": 0.8,
        "iptm": 0.95,
        "sample_ranking_score": 0.9,
        "chain_pair_iptm": {
            f"({a}, {b})": float(pairs[i, j])
            for i, a in enumerate("ABC")
            for j, b in enumerate("ABC")
            if i != j
        },
    }
    pae = [[0, 3, 4, 5], [2, 0, 3, 7], [3, 2, 0, 9], [6, 8, 10, 0]]
    native = of3.compute_metrics(
        summary,
        {"plddt": [80] * 4, "pae": pae},
        structure,
        [
            {"chain_ids": [c], "sequence": seq}
            for c, seq in zip("ABC", ["A", "GG", "A"])
        ],
        "C",
    )
    af3 = compute_af3_metrics(
        {"ptm": 0.8, "iptm": 0.95, "chain_iptm": [0.1, 0.1, expected]},
        {
            "atom_plddts": [80] * 4,
            "atom_chain_ids": ["A", "B", "B", "C"],
            "token_chain_ids": ["A", "B", "B", "C"],
            "pae": pae,
        },
        "C",
    )
    assert native["i_ptm"] == pytest.approx(0.5)
    for metric in (
        "i_ptm",
        "global_i_ptm",
        "ptm",
        "plddt",
        "binder_plddt",
        "i_pae",
        "ipsae_min",
    ):
        assert native[metric] == pytest.approx(af3[metric])
    # Native 0.4.3 emits one key per unordered pair; each value is symmetric.
    summary["chain_pair_iptm"] = {"(A, B)": 0.99, "(A, C)": 0.2, "(B, C)": 0.4}
    native_unordered = of3.compute_metrics(
        summary,
        {"plddt": [80] * 4, "pae": pae},
        structure,
        [
            {"chain_ids": [c], "sequence": seq}
            for c, seq in zip("ABC", ["A", "GG", "A"])
        ],
        "C",
    )
    assert native_unordered["i_ptm"] == pytest.approx(0.3)


def test_missing_duplicate_and_wrong_seed_fail(tmp_path):
    sample(tmp_path, 1)
    with pytest.raises(ValueError, match="Incomplete"):
        of3.collect_output(tmp_path, config(), CHAINS, "B")
    sample(tmp_path, 2, seed=8)
    with pytest.raises(ValueError, match="Incomplete"):
        of3.collect_output(tmp_path, config(), CHAINS, "B")


def test_dimensions_and_sequence_fail(tmp_path):
    sample(tmp_path, 1)
    sample(tmp_path, 2)
    bad = copy.deepcopy(CHAINS)
    bad[0]["sequence"] = "V"
    with pytest.raises(ValueError, match="sequence"):
        of3.collect_output(tmp_path, config(), bad, "B")
    p = tmp_path / "prediction_seed_7_sample_1_confidences.json"
    p.write_text(json.dumps({"plddt": [80, 90], "pae": [[1]]}))
    with pytest.raises(ValueError, match="dimensions"):
        of3.collect_output(tmp_path, config(), CHAINS, "B")


def test_cache_identity_changes_for_software_server_and_chain_order():
    c = config()
    chains = [{"chain_id": "A", "sequence": "AG"}, {"chain_id": "B", "sequence": "GG"}]
    baseline = of3.cache_key(chains, c)
    assert baseline != of3.cache_key(chains[::-1], c)
    assert baseline != of3.cache_key(
        chains, {**c, "msa_server_url": "https://different.test"}
    )
    assert baseline != of3.cache_key(chains, {**c, "software": {"version": "other"}})


def test_runner_disables_nested_parallelism_and_msa():
    settings = of3.runner_settings(config(), Path("/out"))
    assert settings["pl_trainer_args"]["devices"] == 1
    assert settings["pl_trainer_args"]["num_nodes"] == 1
    assert settings["experiment_settings"]["use_msa_server"] is False
    assert settings["experiment_settings"]["use_templates"] is False
    assert settings["experiment_settings"]["seeds"] == [7]
    assert "inference_ckpt_path" not in settings


def test_cli_preserves_default_and_adds_native():
    parser = build_parser()
    for backend in ("af3", "af2", "openfold3"):
        args = parser.parse_args(
            [
                "evaluate",
                "--run-dir",
                "run",
                "--selection",
                "last",
                "--evaluation-name",
                "test",
                "--evaluator-config",
                "config.json",
                "--evaluator",
                backend,
            ]
        )
        assert args.evaluator == backend


def test_resume_checks_content_not_existence(tmp_path):
    artifact = tmp_path / "model.cif"
    artifact.write_text("initial")
    for name in ("confidence.json", "summary.json"):
        (tmp_path / name).write_text("{}")
    p = tmp_path / "result.json"
    p.write_text(
        json.dumps(
            {
                "status": "complete",
                "identity": "same",
                "seeds": [7],
                "num_diffusion_samples": 1,
                "samples": [
                    {
                        "seed": 7,
                        "sample": 0,
                        "structure": "model.cif",
                        "confidences": "confidence.json",
                        "summary_confidences": "summary.json",
                    }
                ],
                "artifacts": {
                    name: of3.file_sha256(tmp_path / name)
                    for name in ("model.cif", "confidence.json", "summary.json")
                },
            }
        )
    )
    expected = {"seeds": [7], "num_diffusion_samples": 1}
    assert of3.complete(p, tmp_path, "same", expected)
    assert not of3.complete(
        p, tmp_path, "same", {**expected, "num_diffusion_samples": 2}
    )
    original = json.loads(p.read_text())
    missing = copy.deepcopy(original)
    del missing["artifacts"]["summary.json"]
    p.write_text(json.dumps(missing))
    assert not of3.complete(p, tmp_path, "same", expected)
    p.write_text(json.dumps(original))
    artifact.write_text("changed")
    assert not of3.complete(p, tmp_path, "same", expected)


def test_evaluation_shards_resume_and_changed_inputs(tmp_path, monkeypatch):
    from odin_multi import atomic_write_json, init_run
    from run_layout import layout_for_run

    target = tmp_path / "target.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C  \nEND\n"
    )
    settings, general, loss = [
        tmp_path / name for name in ("settings.json", "general.json", "loss.json")
    ]
    atomic_write_json(
        settings, {"binder_name": "context", "starting_pdb": str(target), "chains": "A"}
    )
    atomic_write_json(general, {"af_params_dir": str(tmp_path)})
    atomic_write_json(loss, {"role": "target"})
    run = tmp_path / "run"
    init_run(run, [settings], general, [loss], 42)
    layout = layout_for_run(run)
    selection = {
        "status": "complete",
        "rows": [
            {
                "selection_status": "selected",
                "design_index": i,
                "design_id": f"t{i:05d}",
                "sequence": "G",
                "iteration": 3,
                "stage": "hard",
            }
            for i in range(4)
        ],
    }
    atomic_write_json(layout.selection_dir("last") / "selection.json", selection)
    c = config()
    c["timeout_seconds"] = 60
    monkeypatch.setattr(of3, "normalize_config", lambda *args, **kwargs: c)
    monkeypatch.setattr(
        of3,
        "_cached",
        lambda *args: (
            "cache",
            [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "A"}],
        ),
    )
    calls = []

    def infer(config, arguments, work, env):
        assert (
            arguments[arguments.index("--inference-ckpt-path") + 1]
            == config["checkpoint"]
        )
        assert "inference_ckpt_path" not in json.loads(
            (work / "runner.yml").read_text()
        )
        calls.append(work)
        output = work / "outputs"
        output.mkdir()
        sample(output, 1)
        sample(output, 2)

    monkeypatch.setattr(of3, "_run", infer)
    # A concurrent worker holding each job lock owns the work exclusively.
    from contextlib import ExitStack

    from odin_multi import file_lock

    with ExitStack() as stack:
        for i in range(4):
            stack.enter_context(
                file_lock(
                    layout.locks / "openfold3" / "native" / f"t{i:05d}_context.lock"
                )
            )
        busy = of3.run_evaluation(run, "last", "native", config_path=settings)
        assert busy["busy"] == 4
        assert calls == []
    first = of3.run_evaluation(
        run, "last", "native", config_path=settings, shard_index=0, num_shards=2
    )
    second = of3.run_evaluation(
        run, "last", "native", config_path=settings, shard_index=1, num_shards=2
    )
    assert first["completed"] == second["completed"] == 2
    assert of3.collect_evaluation(run, "native") == {"rows": 8, "failed": 0}
    assert (
        of3.run_evaluation(run, "last", "native", config_path=settings)["skipped"] == 4
    )
    assert len(calls) == 4
    # A missing artifact forces precisely its job to rerun.
    victim = next(layout.evaluation_jobs("openfold3", "native").rglob("*_model.cif"))
    victim.unlink()
    assert (
        of3.run_evaluation(run, "last", "native", config_path=settings)["completed"]
        == 1
    )
    selection["rows"][0]["sequence"] = "A"
    atomic_write_json(layout.selection_dir("last") / "selection.json", selection)
    with pytest.raises(ValueError, match="inputs changed"):
        of3.run_evaluation(run, "last", "native", config_path=settings)


def test_environment_python_symlink_is_not_dereferenced(tmp_path):
    import sys

    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    assert of3._path(str(interpreter), tmp_path) == interpreter


def test_cached_alignment_mutation_is_rejected(tmp_path):
    c = config()
    c["target_cache_dir"] = str(tmp_path)
    chains = [{"chain_id": "A", "sequence": "AG"}]
    key = of3.cache_key(chains, c)
    root = tmp_path / key
    root.mkdir()
    alignment = root / "main.a3m"
    alignment.write_text(">query\nAG\n")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "key": key,
                "status": "complete",
                "files": {"main.a3m": of3.file_sha256(alignment)},
                "chains": [
                    {
                        "chain_ids": ["A"],
                        "sequence": "AG",
                        "main_msa_file_paths": ["main.a3m"],
                    }
                ],
            }
        )
    )
    assert of3._cached(chains, c)[1][0]["main_msa_file_paths"] == [str(alignment)]
    metadata_path = root / "metadata.json"
    original = json.loads(metadata_path.read_text())
    changed = copy.deepcopy(original)
    changed["files"] = {}
    metadata_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cache artifact coverage"):
        of3._cached(chains, c)
    metadata_path.write_text(json.dumps(original))
    alignment.write_text(">query\nAA\n")
    with pytest.raises(ValueError, match="cache artifact"):
        of3._cached(chains, c)
