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
