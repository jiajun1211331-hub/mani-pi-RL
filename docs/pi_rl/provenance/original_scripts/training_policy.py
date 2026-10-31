"""Explicit user-selected training budgets, separate from the official recipe."""


def budget_overrides(state: dict) -> dict:
    policy = state.get("training_policy")
    if policy is None:
        return {}
    required = {"outer_steps", "save_interval", "cloud_preflight"}
    if not required <= policy.keys() or set(policy) - required - {"local_lifecycle"}:
        raise ValueError(
            f"Training policy must contain exactly {sorted(required)} and optional local_lifecycle"
        )
    steps, interval = policy["outer_steps"], policy["save_interval"]
    if (
        type(steps) is not int
        or type(interval) is not int
        or not 1 <= interval <= steps
        or policy["cloud_preflight"] not in ("separate", "inline")
        or policy.get("local_lifecycle", "same_snapshot")
        not in ("same_snapshot", "reuse_baseline")
    ):
        raise ValueError("Invalid explicit training budget or cloud preflight mode")
    return {"runner.max_epochs": steps, "runner.save_interval": interval}
