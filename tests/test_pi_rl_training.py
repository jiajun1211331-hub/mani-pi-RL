# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check immutable training admission and frozen-checkpoint contracts."""

import importlib
import socket
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/pi_rl"
sys.path.insert(0, str(SCRIPTS))
prepare = importlib.import_module("prepare_training")
verify = importlib.import_module("verify_training")
submit = importlib.import_module("submit_training")
launch = importlib.import_module("launch_training")
policy = importlib.import_module("training_policy")


def test_explicit_user_budget_does_not_change_smoke_or_algorithm_parameters(tmp_path):
    state = {
        "run_id": "user-budget",
        "sft_path": "/sft",
        "training_policy": {
            "outer_steps": 200,
            "save_interval": 4,
            "cloud_preflight": "inline",
        },
    }
    formal = prepare.phase_overrides(tmp_path, state, "formal")
    assert formal["runner.max_epochs"] == 200
    assert formal["runner.save_interval"] == 4
    assert len(formal) == 10
    finite = prepare.phase_overrides(tmp_path, state, "local-smoke")
    assert finite["runner.max_epochs"] == 1
    assert finite["runner.save_interval"] == 1
    assert not any(key.startswith("algorithm.") for key in formal)
    state["training_policy"]["learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="exactly"):
        policy.budget_overrides(state)


@pytest.mark.parametrize("steps,interval", [(0, 4), (200, 0), (2, 4), (True, 1)])
def test_user_budget_rejects_invalid_step_or_checkpoint_intervals(steps, interval):
    with pytest.raises(ValueError, match="Invalid"):
        policy.budget_overrides(
            {
                "training_policy": {
                    "outer_steps": steps,
                    "save_interval": interval,
                    "cloud_preflight": "inline",
                }
            }
        )


def test_finite_phase_budgets_do_not_enter_formal_overrides(tmp_path):
    state = {"run_id": "test", "sft_path": "/sft"}
    formal = prepare.phase_overrides(tmp_path, state, "formal")
    assert len(formal) == 8
    assert not any(key.startswith("algorithm.") for key in formal)
    for phase, steps in [("local-smoke", 1), ("cloud-preflight", 2)]:
        finite = prepare.phase_overrides(tmp_path, state, phase)
        assert finite["runner.max_epochs"] == steps
        assert "runner.max_epochs" not in formal
        assert finite["runner.logger.log_path"] != formal["runner.logger.log_path"]
        assert "actor.micro_batch_size" not in finite
        assert "actor.global_batch_size" not in finite
        assert "algorithm.group_size" not in finite
    with pytest.raises(ValueError, match="Unknown training phase"):
        prepare.phase_overrides(tmp_path, state, "paper-budget")


def test_ray_authentication_stays_in_private_runtime(tmp_path):
    environment = launch.private_ray_environment(tmp_path)
    token = Path(environment["RAY_AUTH_TOKEN_PATH"])
    assert token.parent == tmp_path
    assert environment["RAY_AUTH_MODE"] == "token"
    assert token.stat().st_mode & 0o777 == 0o600
    original = token.read_bytes()
    assert len(original) == 64
    assert launch.private_ray_environment(tmp_path) == environment
    assert token.read_bytes() == original
    token.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode 0600"):
        launch.private_ray_environment(tmp_path)


def test_short_temp_directory_accepts_real_unix_sockets(tmp_path):
    with tempfile.TemporaryDirectory(prefix="pirl-tmp-", dir="/dev/shm") as root:
        short = launch.short_temporary_directory(tmp_path, Path(root))
        assert short.resolve() == tmp_path / "tmp"
        address = short / "socket-probe"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(address))
            assert address.exists()
        address.unlink()


def test_torchrun_does_not_parse_the_probe_run_argument_as_its_own_option():
    pytest.importorskip("torch")
    from torch.distributed.run import get_args_parser

    command = launch.collective_probe_command(
        Path("/env/bin/python"), Path("/snapshot"), Path("/run"), "local-smoke", 2
    )
    parsed = get_args_parser().parse_args(command[3:])
    assert parsed.training_script == "/snapshot/harness/gpu_collective_probe.py"
    assert parsed.training_script_args == ["--run", "/run", "--phase", "local-smoke"]


def test_training_entrypoint_quotes_unicode_and_spaces_for_hydra():
    pytest.importorskip("hydra")
    from hydra.core.override_parser.overrides_parser import OverridesParser

    private = Path("/vepfs-C区/run with spaces/runtime")
    command = launch.training_entrypoint_command(
        Path("/env/bin/python"),
        Path("/snapshot"),
        Path("/configs/local-smoke.yaml"),
        private,
    )
    parsed = OverridesParser.create().parse_overrides(command[-3:])
    values = {item.key_or_group: item.value() for item in parsed}
    assert values["hydra.run.dir"] == str(private / "work")
    assert values["hydra.output_subdir"] is None
    assert values["hydra.job.chdir"] is False


def test_validation_reads_tensorboard_beside_experiment_checkpoints(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    (tmp_path / "experiment/checkpoints/global_step_1").mkdir(parents=True)
    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        writer.add_scalar("eval/success_once", 1.0, 0)
        writer.add_scalar("eval/num_trajectories", 2, 0)
        writer.add_scalar("actor/loss", 0.1, 0)
    result = verify.read_validation_metrics(tmp_path)
    assert result == {
        "eval/success_once": [{"step": 0, "value": 1.0}],
        "eval/num_trajectories": [{"step": 0, "value": 2.0}],
    }


def test_volc_banner_does_not_hide_existing_tasks_or_create_false_matches():
    row = {"JobId": "t-example", "JobName": "spatial-formal", "Status": "Running"}
    output = "[Metrics(v3)] unavailable\n" + __import__("json").dumps([row])
    rows = submit.parse_task_list(output)
    assert submit.exact_task(rows, "spatial-formal") == row
    assert submit.exact_task(rows, "spatial") is None
    with pytest.raises(RuntimeError, match="Multiple"):
        submit.exact_task(rows + rows, "spatial-formal")
    with pytest.raises(RuntimeError, match="no valid JSON"):
        submit.parse_task_list('[Metrics(v3)] error\n[{"error": "query failed"}]')


@pytest.mark.parametrize("mutation", ["content", "added", "missing", "symlink"])
def test_frozen_source_rejects_mutation(tmp_path, mutation):
    source = tmp_path / "model.py"
    source.write_text("initial")
    link = tmp_path / "config"
    link.symlink_to("model.py")
    files = {
        str(p.relative_to(tmp_path)): prepare.file_identity(p)
        for p in tmp_path.iterdir()
    }
    prepare.verify_snapshot(tmp_path, files)
    if mutation == "content":
        source.write_text("changed")
    elif mutation == "added":
        (tmp_path / "custom_loss.py").write_text("extra")
    elif mutation == "missing":
        source.unlink()
    else:
        link.unlink()
        link.symlink_to("other.py")
    with pytest.raises(RuntimeError, match="Frozen training source changed"):
        prepare.verify_snapshot(tmp_path, files)


@pytest.mark.parametrize(
    "mutation", ["valid", "frozen", "nonfinite", "missing", "unchanged"]
)
def test_strict_reload_requires_frozen_vlm_and_real_actor_change(mutation):
    torch = pytest.importorskip("torch")
    model = torch.nn.ModuleDict(
        {"vlm": torch.nn.Linear(2, 2), "actor": torch.nn.Linear(2, 1)}
    )
    model["vlm"].requires_grad_(False)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    if mutation != "unchanged":
        weights["actor.weight"] += 1
    if mutation == "frozen":
        weights["vlm.weight"] += 1
    elif mutation == "nonfinite":
        weights["actor.weight"][0, 0] = float("nan")
    elif mutation == "missing":
        del weights["vlm.bias"]
    if mutation == "valid":
        report = verify.compare_weights(model, weights)
        assert report["passed"]
        assert report["missing_keys"] == []
        assert report["frozen_tensors_unchanged"] == 2
    else:
        with pytest.raises(RuntimeError, match="Checkpoint"):
            verify.compare_weights(model, weights)


@pytest.fixture
def mani_source_repo(tmp_path):
    import subprocess

    from scripts.pi_rl.mani_v1.prepare_training import REQUIRED

    repo = tmp_path / "source repo"
    repo.mkdir()
    for name in REQUIRED:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# committed source\n")
    (repo / "rlinf/obsolete.py").write_text("# removable source\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Snapshot Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )
    return repo


def test_mani_development_snapshot_includes_modified_and_new_runtime_source(
    mani_source_repo, tmp_path
):
    from scripts.pi_rl.mani_v1.prepare_training import freeze_source

    modified = mani_source_repo / "rlinf/algorithms/manifold/metrics.py"
    modified.write_text("# actual modified metrics\n")
    added = mani_source_repo / "rlinf/algorithms/manifold/new_runtime.py"
    added.write_text("# new runtime file\n")
    excluded = mani_source_repo / "docs/research.md"
    excluded.parent.mkdir()
    excluded.write_text("nonruntime evidence")
    run = tmp_path / "development"
    with pytest.raises(RuntimeError, match="clean committed source"):
        freeze_source(mani_source_repo, run)
    assert not run.exists()
    manifest = freeze_source(mani_source_repo, run, allow_dirty=True)
    assert manifest["development_snapshot"]
    for path in (modified, added):
        assert (
            run / "source" / path.relative_to(mani_source_repo)
        ).read_bytes() == path.read_bytes()
    assert "docs/research.md" in manifest["excluded_nonruntime_paths"]
    assert "new_runtime.py" in manifest["git_status"]
    assert b"actual modified metrics" in (run / "manifests/worktree.diff").read_bytes()
    assert manifest["commits"][0].endswith("test fixture")
    with pytest.raises(FileExistsError):
        freeze_source(mani_source_repo, run, allow_dirty=True)


def test_mani_clean_snapshot_preserves_committed_source_and_deletions(
    mani_source_repo, tmp_path
):
    from scripts.pi_rl.mani_v1.prepare_training import freeze_source

    manifest = freeze_source(mani_source_repo, tmp_path / "clean")
    assert not manifest["development_snapshot"]
    assert manifest["git_status"] == ""
    obsolete = mani_source_repo / "rlinf/obsolete.py"
    obsolete.unlink()
    development = freeze_source(
        mani_source_repo, tmp_path / "deleted", allow_dirty=True
    )
    assert "rlinf/obsolete.py" not in development["files"]
    assert not (tmp_path / "deleted/source/rlinf/obsolete.py").exists()
    deleted = mani_source_repo / "scripts/pi_rl/mani_v1/launch_training.py"
    deleted.unlink()
    with pytest.raises(ValueError, match="Missing mani_v1"):
        freeze_source(mani_source_repo, tmp_path / "missing", allow_dirty=True)


def test_mani_source_rejects_external_symlinks(mani_source_repo, tmp_path):
    from scripts.pi_rl.mani_v1.prepare_training import freeze_source

    (mani_source_repo / "rlinf/external.py").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="internal relative source symlink"):
        freeze_source(mani_source_repo, tmp_path / "run", allow_dirty=True)


@pytest.fixture(scope="module")
def prepared_mani_run(tmp_path_factory):
    from scripts.pi_rl.mani_v1.prepare_training import prepare as prepare_mani

    base = tmp_path_factory.mktemp("mani-runtime")
    model = base / "model with spaces"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"identity-only model fixture")
    run = base / "run with spaces"
    prepare_mani(
        SCRIPTS.parents[1],
        run,
        "libero_10_grpo_openpi_pi05",
        [],
        model,
        allow_dirty=True,
    )
    from omegaconf import OmegaConf

    resolved = OmegaConf.load(run / "configs/train.yaml")
    assert Path(resolved.runner.logger.log_path).is_relative_to(run)
    assert resolved.runner.logger.experiment_name == run.name
    assert Path(resolved.actor.model.model_path) == model
    return run


@pytest.fixture
def copied_mani_run(prepared_mani_run, tmp_path):
    import shutil

    run = tmp_path / "copied run"
    shutil.copytree(prepared_mani_run, run)
    return run


@pytest.mark.parametrize(
    "mutation", ["content", "added", "missing", "symlink", "config"]
)
def test_mani_snapshot_detects_mutation(copied_mani_run, mutation):
    from rlinf.algorithms.manifold.snapshot import verify_snapshot

    run = copied_mani_run
    verify_snapshot(run)
    source = run / "source/rlinf/algorithms/manifold/metrics.py"
    if mutation == "content":
        source.write_text("changed")
    elif mutation == "added":
        source.with_name("injected.py").write_text("extra")
    elif mutation == "missing":
        source.unlink()
    elif mutation == "config":
        (run / "configs/train.yaml").write_text("changed")
    else:
        source.unlink()
        source.symlink_to("groups.py")
    with pytest.raises(RuntimeError, match="changed"):
        verify_snapshot(run)


def test_mani_runtime_records_loaded_origins_and_rejects_workspace_imports(
    prepared_mani_run,
):
    import json
    import os
    import subprocess

    run = prepared_mani_run
    code = "import json; import rlinf.config; import rlinf.algorithms.manifold.metrics; from rlinf.algorithms.manifold.snapshot import verify_runtime_source; print(json.dumps(verify_runtime_source('test')))"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(run / "source"),
            "RLINF_MANIFOLD_RUN": str(run),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=run,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    for name in ("rlinf.config", "rlinf.algorithms.manifold.metrics"):
        assert Path(report["modules"][name]["file"]).is_relative_to(run / "source")
    assert list((run / "runtime/source_reports").glob("test_*.json"))
    environment["PYTHONPATH"] = str(SCRIPTS.parents[1])
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=run,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "from frozen source" in result.stderr


def test_mani_prepared_launcher_checks_without_starting_training(prepared_mani_run):
    import os
    import subprocess

    run = prepared_mani_run
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(run / "source"), "PYTHONDONTWRITEBYTECODE": "1"})
    subprocess.run(
        [
            sys.executable,
            str(run / "source/scripts/pi_rl/mani_v1/launch_training.py"),
            "--run",
            str(run),
            "--check-only",
        ],
        cwd=run,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (run / "runtime/launch.json").exists()
    assert list((run / "runtime/source_reports").glob("launch_*.json"))
    assert not (run / "source/.git").exists()


def test_mani_launcher_rejects_changed_external_model(copied_mani_run, tmp_path):
    import json

    from rlinf.algorithms.manifold.snapshot import sha256
    from scripts.pi_rl.mani_v1.launch_training import verify_run

    run = copied_mani_run
    model = tmp_path / "new-model"
    model.mkdir()
    original = json.loads((run / "manifests/model.json").read_text())
    original["path"] = str(model)
    (model / "model.safetensors").write_bytes(b"identity-only model fixture")
    (run / "manifests/model.json").write_text(json.dumps(original))
    manifest = json.loads((run / "manifests/source.json").read_text())
    manifest["model_manifest_sha256"] = sha256(run / "manifests/model.json")
    (run / "manifests/source.json").write_text(json.dumps(manifest))
    verify_run(run)
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="model changed"):
        verify_run(run)


@pytest.mark.parametrize("key", ["actor.model.model_path", "runner.logger.log_path"])
def test_mani_prepare_rejects_conflicting_owned_paths(mani_source_repo, tmp_path, key):
    from scripts.pi_rl.mani_v1.prepare_training import prepare as prepare_mani

    model = tmp_path / "model"
    model.mkdir()
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="owned by --run/--model"):
        prepare_mani(mani_source_repo, run, "recipe", [f"{key}=elsewhere"], model)
    assert not run.exists()


def test_mani_native_hydra_arguments_preserve_unicode_and_spaces():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from scripts.pi_rl.mani_v1.launch_training import training_arguments

    run = Path("/vepfs-C区/run with spaces")
    args = training_arguments(run)
    parsed = OverridesParser.create().parse_overrides(args[-3:])
    values = {item.key_or_group: item.value() for item in parsed}
    assert values["hydra.run.dir"] == str(run / "runtime/work")
    assert values["hydra.output_subdir"] is None
    assert values["hydra.job.chdir"] is False
