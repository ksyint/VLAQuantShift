"""Validated factorial plans. Commands are argument arrays, never shell programs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import UPSTREAM_COMMIT


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _int(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _keys(mapping, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


def validate_config(config):
    """Reject ambiguous splits, duplicate cells and unsupported experimental knobs."""
    from vlaquantbench.quant.presets import parse_spec
    from vlaquantbench.registry import BENCHMARKS, MODELS
    from .shifts import ShiftSpec

    allowed = ("schema_version", "model", "checkpoint", "benchmark", "suite", "dtype", "device", "tasks",
               "seeds", "shift_seed", "episodes", "calibration", "conditions", "quantizations", "max_steps")
    _keys(config, allowed, "config")
    cfg = json.loads(json.dumps(config, allow_nan=False))
    if type(cfg.get("schema_version")) is not int or cfg["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for field in ("model", "checkpoint", "benchmark", "suite"):
        if not isinstance(cfg.get(field), str) or not cfg[field].strip():
            raise ValueError(f"{field} is required")
    if cfg["model"] == "fixture_controller":
        from .fixture import register_fixture
        register_fixture()
        if cfg["benchmark"] != "fixture_reach" or cfg["checkpoint"] != "builtin-fixed-v1":
            raise ValueError("synthetic fixture requires fixture_reach and builtin-fixed-v1")
    if cfg["model"] not in MODELS or cfg["model"] == "remote":
        raise ValueError("plan requires a local registered model; remote quantization is not audited here")
    if cfg["benchmark"] not in BENCHMARKS:
        raise ValueError("unknown benchmark")
    if cfg["checkpoint"] == "auto":
        raise ValueError("pin an explicit checkpoint identifier in the plan")
    cfg.setdefault("dtype", "bf16")
    from vlaquantbench.models.base import resolve_dtype
    resolve_dtype(cfg["dtype"])
    cfg.setdefault("device", "cuda:0")
    if not isinstance(cfg["device"], str) or not (cfg["device"] == "cpu" or cfg["device"].startswith("cuda")):
        raise ValueError("device must be cpu or cuda[:index]")
    for field in ("tasks", "seeds"):
        values = cfg.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"explicit nonempty {field} required for coverage auditing")
        for v in values:
            _int(v, field)
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field}")
        cfg[field] = sorted(values)
    cfg.setdefault("shift_seed", 0)
    _int(cfg["shift_seed"], "shift_seed")
    ep = cfg.get("episodes", {})
    _keys(ep, ("start", "stop"), "episodes")
    start, stop = _int(ep.get("start"), "episodes.start"), _int(ep.get("stop"), "episodes.stop", 1)
    if stop <= start:
        raise ValueError("episodes.stop must exceed start (exclusive endpoint)")
    if cfg.get("max_steps") is not None:
        _int(cfg["max_steps"], "max_steps", 1)
    cfg.setdefault("max_steps", None)
    conditions = cfg.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 2:
        raise ValueError("conditions must contain clean and at least one shifted condition")
    cfg["conditions"] = [ShiftSpec.parse(x).canonical for x in conditions]
    if "clean" not in cfg["conditions"] or len(set(cfg["conditions"])) != len(cfg["conditions"]):
        raise ValueError("conditions must contain one clean condition and no duplicates")
    cal = cfg.setdefault("calibration", {"components": [], "shift": "clean", "start": stop, "count": 2, "tasks": 1})
    _keys(cal, ("components", "shift", "start", "count", "tasks", "alpha", "quantile"), "calibration")
    cal.setdefault("components", [])
    if not isinstance(cal["components"], list) or any(x not in ("ve", "mp", "llm", "ah") for x in cal["components"]):
        raise ValueError("invalid calibration components")
    if len(set(cal["components"])) != len(cal["components"]):
        raise ValueError("duplicate calibration components")
    cal["shift"] = ShiftSpec.parse(cal.get("shift", "clean")).canonical
    cal.setdefault("start", stop)
    cal.setdefault("count", 2)
    cal.setdefault("tasks", 1)
    _int(cal["start"], "calibration.start")
    _int(cal["count"], "calibration.count", 1)
    _int(cal["tasks"], "calibration.tasks", 1)
    if cal["components"] and max(start, cal["start"]) < min(stop, cal["start"] + cal["count"]):
        raise ValueError("calibration and evaluation episode indices overlap")
    cal.setdefault("alpha", 0.5)
    cal.setdefault("quantile", 0.999)
    if not 0 <= cal["alpha"] <= 1 or not 0 < cal["quantile"] <= 1:
        raise ValueError("calibration alpha/quantile out of range")
    quantizations = cfg.get("quantizations")
    if not isinstance(quantizations, list) or not quantizations:
        raise ValueError("at least one quantization is required")
    names = set()
    settings = set()
    for q in quantizations:
        _keys(q, ("name", "preset", "scope", "exclude"), "quantization")
        name = q.get("name")
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("quantization name must use letters, digits, underscore or hyphen")
        if name in names or name == "baseline":
            raise ValueError("quantization names must be unique and cannot be baseline")
        names.add(name)
        spec = parse_spec(q.get("preset", ""))
        if spec.is_baseline:
            raise ValueError("baseline is automatically included; quantizations must change precision")
        if cal["components"] and spec.act is None:
            raise ValueError("activation calibration requires an activation-quantized preset, e.g. W4A8")
        q["preset"] = spec.name
        q.setdefault("scope", "e2e")
        if q["scope"] != "e2e" and any(c not in ("ve", "mp", "llm", "ah") for c in q["scope"].split("+")):
            raise ValueError("invalid component scope")
        q.setdefault("exclude", [])
        if not isinstance(q["exclude"], list) or any(not isinstance(x, str) or not x for x in q["exclude"]):
            raise ValueError("exclude must be a list of nonempty glob strings")
        setting = digest({k: v for k, v in q.items() if k != "name"})
        if setting in settings:
            raise ValueError("duplicate quantization settings under different names")
        settings.add(setting)
    return cfg


def build_plan(config, output_root="shift_runs"):
    cfg = validate_config(config)
    config_hash = digest(cfg)
    root = Path(output_root) / config_hash[:12]
    jobs, comparisons = [], []
    by_key = {}
    baseline = {"name": "baseline", "preset": "BASELINE", "scope": "e2e", "exclude": []}
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            for quant in [baseline] + cfg["quantizations"]:
                cell = {"config_sha256": config_hash, "seed": seed, "condition": condition, "quantization": quant["name"]}
                cell_id = digest(cell)[:16]
                out = str(root / f"{cell_id}.jsonl")
                module = "vlaquantshift.fixture" if cfg["model"] == "fixture_controller" else "vlaquantbench.cli"
                argv = ["python", "-m", module, "run", "--model", cfg["model"],
                        "--checkpoint", cfg["checkpoint"], "--benchmark", cfg["benchmark"], "--suite", cfg["suite"],
                        "--dtype", cfg["dtype"], "--device", cfg["device"], "--preset", quant["preset"],
                        "--scope", quant["scope"], "--method", "rtn", "--seed", str(seed),
                        "--episodes-from", str(cfg["episodes"]["start"]), "--episodes", str(cfg["episodes"]["stop"]),
                        "--tasks", ",".join(map(str, cfg["tasks"])), "--eval-shift", condition,
                        "--calib-shift", cfg["calibration"]["shift"], "--shift-seed", str(cfg["shift_seed"]), "--out", out]
                if cfg["max_steps"] is not None:
                    argv += ["--max-steps", str(cfg["max_steps"])]
                if quant["exclude"]:
                    argv += ["--exclude"] + quant["exclude"]
                cal = cfg["calibration"]
                if cal["components"] and quant["name"] != "baseline":
                    argv += ["--act-calib"] + cal["components"]
                    for key, flag in (("start", "from"), ("count", "episodes"), ("tasks", "tasks"), ("alpha", "alpha"), ("quantile", "quantile")):
                        argv += [f"--act-calib-{flag}", str(cal[key])]
                jobs.append({"id": cell_id, **cell, "out": out, "argv": argv})
                by_key[seed, condition, quant["name"]] = out
        for condition in cfg["conditions"]:
            if condition == "clean":
                continue
            for quant in cfg["quantizations"]:
                comparisons.append({"seed": seed, "condition": condition, "quantization": quant["name"],
                                    "fp_clean": by_key[seed, "clean", "baseline"],
                                    "q_clean": by_key[seed, "clean", quant["name"]],
                                    "fp_shift": by_key[seed, condition, "baseline"],
                                    "q_shift": by_key[seed, condition, quant["name"]]})
    plan = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT, "config": cfg,
            "config_sha256": config_hash, "output_root": str(output_root), "jobs": jobs, "comparisons": comparisons,
            "total_episodes": len(jobs) * len(cfg["tasks"]) * (cfg["episodes"]["stop"] - cfg["episodes"]["start"])}
    plan["plan_sha256"] = digest(plan)
    return plan


def verify_plan(plan):
    rebuilt = build_plan(plan["config"], plan["output_root"])
    if plan != rebuilt:
        raise ValueError("plan differs from its validated configuration; regenerate it")
    return plan


def execute_plan(plan, *, gpu_ids="6,7", job_ids=None):
    """Run one cell per child process; callers choose the installed model environment."""
    verify_plan(plan)
    selected = set(job_ids) if job_ids else {j["id"] for j in plan["jobs"]}
    unknown = selected - {j["id"] for j in plan["jobs"]}
    if unknown:
        raise ValueError(f"unknown job ids: {sorted(unknown)}")
    if not gpu_ids or any(not part.isdecimal() for part in gpu_ids.split(",")):
        raise ValueError("GPU ids must be a comma-separated list of physical device indices")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if plan["config"]["device"] == "cpu" else gpu_ids
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for job in plan["jobs"]:
        if job["id"] in selected:
            subprocess.run([sys.executable, *job["argv"][1:]], env=env, check=True)
