"""Strict, paired analyses for quantization under observation shift.

The success-rate interaction is descriptive: it measures how the quantization
penalty changes under a specified intervention on observations. It does not
identify a physical mechanism or a causal mediation effect.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LoadedRun", "load_run", "analyze_factorial", "analyze_cross_replay"]

EpisodeKey = tuple[str, int, int, int]
ARM_NAMES = ("fp_clean", "q_clean", "fp_shift", "q_shift")


@dataclass(frozen=True)
class LoadedRun:
    """A validated JSONL run; episode keys include the actual episode seed."""

    header: dict[str, Any]
    episodes: dict[EpisodeKey, dict[str, Any]]
    path: str


def _object_no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON member {key!r}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number {value}")


def _finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: nonfinite numeric value")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item, label)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _string(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or (not empty and not value):
        raise ValueError(f"{label} must be a {'possibly empty ' if empty else ''}trimmed string")
    return value


def _protocol(header: dict[str, Any]) -> dict[str, Any]:
    extra = header.get("extra")
    if not isinstance(extra, dict) or not isinstance(extra.get("vlaquantshift"), dict):
        raise ValueError("missing extra.vlaquantshift protocol metadata")
    return extra["vlaquantshift"]


def _validate_header(header: dict[str, Any]) -> None:
    if header.get("kind") != "header":
        raise ValueError("first JSONL record must have kind='header'")
    for field in ("model", "checkpoint", "benchmark", "suite", "preset", "scope", "method", "baseline_dtype"):
        _string(header.get(field), f"header.{field}")
    _integer(header.get("seed"), "header.seed")
    _integer(header.get("episodes_per_task"), "header.episodes_per_task", 1)
    for field in ("quantize_lm_head", "include_conv"):
        if type(header.get(field)) is not bool:
            raise ValueError(f"header.{field} must be a bool")
    if not isinstance(header.get("quant_report"), dict):
        raise ValueError("header.quant_report must be an object")
    protocol = _protocol(header)
    if type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1:
        raise ValueError("extra.vlaquantshift.schema_version must be 1")
    if "baseline_dtype" in protocol and protocol["baseline_dtype"] != header["baseline_dtype"]:
        raise ValueError("protocol baseline_dtype differs from header baseline_dtype")
    from .shifts import ShiftSpec

    for field in ("eval_shift", "calib_shift"):
        value = _string(protocol.get(field), f"vlaquantshift.{field}")
        if ShiftSpec.parse(value).canonical != value:
            raise ValueError(f"vlaquantshift.{field} must use canonical shift syntax")
    _integer(protocol.get("shift_seed"), "vlaquantshift.shift_seed")
    start = _integer(protocol.get("episodes_from"), "vlaquantshift.episodes_from")
    stop = _integer(protocol.get("episodes_to"), "vlaquantshift.episodes_to", 1)
    if not start < stop <= header["episodes_per_task"]:
        raise ValueError("episode interval must satisfy 0 <= episodes_from < episodes_to <= episodes_per_task")
    if "tasks" not in protocol or "max_steps" not in protocol:
        raise ValueError("protocol must explicitly record tasks and max_steps (null is allowed)")
    if protocol["max_steps"] is not None:
        _integer(protocol["max_steps"], "vlaquantshift.max_steps", 1)
    if protocol["tasks"] is not None:
        text = _string(protocol["tasks"], "vlaquantshift.tasks")
        tokens = text.split(",")
        if any(not token.isdecimal() or str(int(token)) != token for token in tokens):
            raise ValueError("vlaquantshift.tasks must be comma-separated nonnegative task IDs")
        if len(set(tokens)) != len(tokens):
            raise ValueError("vlaquantshift.tasks contains duplicate task IDs")
    if "episodes_by_task" in protocol:
        planned = protocol["episodes_by_task"]
        if not isinstance(planned, dict) or not planned:
            raise ValueError("vlaquantshift.episodes_by_task must be a nonempty object")
        for task_id, indices in planned.items():
            if not isinstance(task_id, str) or not task_id.isdecimal() or str(int(task_id)) != task_id:
                raise ValueError("episodes_by_task keys must be canonical nonnegative task IDs")
            if not isinstance(indices, list) or not indices:
                raise ValueError("episodes_by_task values must be nonempty lists of episode indices")
            for index in indices:
                _integer(index, "episodes_by_task episode index")
            if indices != sorted(set(indices)) or any(index < start or index >= stop for index in indices):
                raise ValueError("episodes_by_task indices must be sorted, unique and inside the episode interval")
        if protocol["tasks"] is not None and set(planned) != set(protocol["tasks"].split(",")):
            raise ValueError("episodes_by_task task IDs differ from declared tasks")
    _finite_json(header, "header")


def _episode_key(record: dict[str, Any], header: dict[str, Any]) -> EpisodeKey:
    if record.get("kind") != "episode":
        raise ValueError("every record after the header must have kind='episode'")
    suite = _string(record.get("suite"), "episode.suite")
    if suite != header["suite"]:
        raise ValueError("episode suite differs from header suite")
    task_id = _integer(record.get("task_id"), "episode.task_id")
    episode = _integer(record.get("episode"), "episode.episode")
    seed = _integer(record.get("seed"), "episode.seed")
    if type(record.get("success")) is not bool:
        raise ValueError("episode.success must be a bool, not a numeric or string value")
    _string(record.get("task_name"), "episode.task_name", empty=True)
    if "steps" in record:
        _integer(record["steps"], "episode.steps")
    if "extra" in record and not isinstance(record["extra"], dict):
        raise ValueError("episode.extra must be an object")
    if "vlaquantshift" in record.get("extra", {}):
        metadata = record["extra"]["vlaquantshift"]
        if not isinstance(metadata, dict):
            raise ValueError("episode.extra.vlaquantshift must be an object")
        protocol = _protocol(header)
        if type(metadata.get("schema_version")) is not int or type(metadata.get("shift_seed")) is not int:
            raise ValueError("episode schema_version and shift_seed must be integers")
        _equal_fields(protocol, metadata, ("schema_version", "eval_shift", "shift_seed"), "episode shift metadata")
        if type(metadata.get("policy_seed")) is not int or metadata["policy_seed"] != seed:
            raise ValueError("episode policy_seed differs from episode seed")
    _finite_json(record, "episode")
    return suite, task_id, episode, seed


def _validate_run(run: LoadedRun) -> None:
    _validate_header(run.header)
    if not run.episodes:
        raise ValueError(f"{run.path}: no episodes")
    protocol = _protocol(run.header)
    expected_episodes = set(range(protocol["episodes_from"], protocol["episodes_to"]))
    by_task = defaultdict(set)
    names = {}
    seen_positions = set()
    for key, record in run.episodes.items():
        if _episode_key(record, run.header) != key:
            raise ValueError(f"{run.path}: episode dictionary key differs from record")
        position = key[:3]
        if position in seen_positions:
            raise ValueError(f"{run.path}: duplicate episode position with conflicting seed: {position}")
        seen_positions.add(position)
        task = key[:2]
        by_task[task].add(key[2])
        if task in names and names[task] != record["task_name"]:
            raise ValueError(f"{run.path}: inconsistent task_name for {task}")
        names[task] = record["task_name"]
    planned = protocol.get("episodes_by_task")
    if planned is not None and {str(task_id) for _, task_id in by_task} != set(planned):
        raise ValueError(f"{run.path}: task coverage differs from episodes_by_task")
    for task, observed in by_task.items():
        expected = expected_episodes if planned is None else set(planned[str(task[1])])
        if observed != expected:
            raise ValueError(
                f"{run.path}: episode coverage differs from declared interval for {task}; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
    if protocol["tasks"] is not None:
        expected_tasks = {int(token) for token in protocol["tasks"].split(",")}
        if {task_id for _, task_id in by_task} != expected_tasks:
            raise ValueError(f"{run.path}: task coverage differs from declared tasks")


def load_run(path: str | os.PathLike) -> LoadedRun:
    """Read one complete run, rejecting corruption instead of silently dropping it.

    Unlike the upstream resumable reader, this analysis reader rejects truncated
    lines, duplicate JSON members, repeated headers, nonfinite numbers, missing
    outcomes and incomplete declared episode ranges.
    """
    path = Path(path)
    header = None
    episodes = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                obj = json.loads(line, object_pairs_hook=_object_no_duplicates, parse_constant=_reject_constant)
                if not isinstance(obj, dict):
                    raise ValueError("record must be a JSON object")
                if header is None:
                    _validate_header(obj)
                    header = obj
                else:
                    key = _episode_key(obj, header)
                    if key in episodes:
                        raise ValueError(f"duplicate episode key {key}")
                    episodes[key] = obj
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if header is None:
        raise ValueError(f"{path}: empty run (missing header)")
    run = LoadedRun(header=header, episodes=episodes, path=str(path))
    _validate_run(run)
    return run


def _equal_fields(left: dict, right: dict, fields, label: str) -> None:
    for field in fields:
        if (field in left) != (field in right) or left.get(field) != right.get(field):
            raise ValueError(f"{label}: mismatched {field}")


def _applied_layer_count(header: dict[str, Any]) -> int:
    """Read upstream component reports or its method-level layer count."""
    report = header["quant_report"]
    if "layers" in report:
        return _integer(report["layers"], "quant_report.layers")
    return sum(
        _integer(component["layers"], f"quant_report.{name}.layers")
        for name, component in report.items()
        if name != "act_calib" and isinstance(component, dict) and "layers" in component
    )


def _calibration_hash(header: dict[str, Any]) -> str | None:
    """Require calibration provenance when calibration is declared in a Q arm."""
    extra = header["extra"]
    report = header["quant_report"]
    indicated = bool(extra.get("act_calib") or extra.get("calib") or extra.get("act_calib_config") or report.get("act_calib"))
    for name, component in report.items():
        if name != "act_calib" and isinstance(component, dict) and "calibrated_layers" in component:
            indicated |= _integer(component["calibrated_layers"], f"quant_report.{name}.calibrated_layers") > 0
    hashes = []
    for provenance in (extra.get("act_calib_config"), report.get("act_calib")):
        if provenance is None:
            continue
        if not isinstance(provenance, dict):
            raise ValueError("calibration provenance must be an object")
        digest = provenance.get("cache_sha256")
        if digest is not None:
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ValueError("calibration cache_sha256 must be a 64-character hexadecimal digest")
            hashes.append(digest.lower())
    if indicated and not hashes:
        raise ValueError("calibration is indicated but no calibration cache_sha256 is recorded")
    if len(set(hashes)) > 1:
        raise ValueError("calibration cache_sha256 differs between header provenance records")
    return hashes[0] if hashes else None


def _matched_runs(inputs) -> list[LoadedRun]:
    runs = [value if isinstance(value, LoadedRun) else load_run(value) for value in inputs]
    for run in runs:
        _validate_run(run)
    reference = runs[0]
    common = (
        "model", "checkpoint", "benchmark", "suite", "baseline_dtype", "seed", "episodes_per_task",
        "vqb_commit", "code_revision",
    )
    protocol_common = (
        "schema_version", "shift_seed", "calib_shift", "episodes_from", "episodes_to", "tasks", "max_steps",
        "episodes_by_task", "episode_rng", "protocol_version", "code_revision",
    )
    for name, run in zip(ARM_NAMES[1:], runs[1:]):
        _equal_fields(reference.header, run.header, common, name)
        _equal_fields(_protocol(reference.header), _protocol(run.header), protocol_common, name)
        reference_protocol = {key: value for key, value in _protocol(reference.header).items() if key != "eval_shift"}
        run_protocol = {key: value for key, value in _protocol(run.header).items() if key != "eval_shift"}
        if reference_protocol != run_protocol:
            raise ValueError(f"{name}: mismatched protocol metadata outside eval_shift")
        _equal_fields(
            reference.header["extra"], run.header["extra"],
            ("adapter_options", "runner_options"), f"{name} execution options",
        )
        if set(reference.episodes) != set(run.episodes):
            missing = len(set(reference.episodes) - set(run.episodes))
            extra = len(set(run.episodes) - set(reference.episodes))
            raise ValueError(f"{name}: exact episode/seed pairing required; missing={missing}, extra={extra}")
        for key in reference.episodes:
            if reference.episodes[key]["task_name"] != run.episodes[key]["task_name"]:
                raise ValueError(f"{name}: mismatched task_name for {key[:2]}")
    for index in (0, 2):
        if runs[index].header["preset"] != "BASELINE":
            raise ValueError(f"{ARM_NAMES[index]} must use preset BASELINE")
        if _applied_layer_count(runs[index].header) != 0:
            raise ValueError(f"{ARM_NAMES[index]} claims BASELINE but reports applied quantized layers")
    for index in (1, 3):
        if runs[index].header["preset"].upper() in {"BASELINE", "FP", "BF16", "FP32", "FP16", "NONE", "W16"}:
            raise ValueError(f"{ARM_NAMES[index]} must use a non-BASELINE quantization preset")
        if _applied_layer_count(runs[index].header) <= 0:
            raise ValueError(f"{ARM_NAMES[index]} must report a positive applied quantized layer count")
    calibration_hashes = [_calibration_hash(runs[index].header) for index in (1, 3)]
    if calibration_hashes[0] != calibration_hashes[1]:
        raise ValueError("quantized arms: mismatched calibration cache_sha256")
    for index in (0, 1):
        if _protocol(runs[index].header)["eval_shift"] != "clean":
            raise ValueError(f"{ARM_NAMES[index]} must use eval_shift='clean'")
    eval_shift = _protocol(runs[2].header)["eval_shift"]
    if eval_shift == "clean" or eval_shift != _protocol(runs[3].header)["eval_shift"]:
        raise ValueError("shift arms must use the same nonclean eval_shift")
    quant_fields = ("preset", "scope", "method", "quantize_lm_head", "include_conv", "quant_report")
    _equal_fields(runs[0].header, runs[2].header, quant_fields, "FP arms")
    _equal_fields(runs[1].header, runs[3].header, quant_fields, "quantized arms")
    # Calibration metadata must match across evaluation conditions. Metadata may
    # contain calibration data paths, activation settings, masks, or fingerprints;
    # preserve all of it except the intentional eval_shift contrast.
    for first, second, label in ((0, 2, "FP arms"), (1, 3, "quantized arms")):
        left_extra = dict(runs[first].header["extra"])
        right_extra = dict(runs[second].header["extra"])
        left_protocol = dict(left_extra.pop("vlaquantshift"))
        right_protocol = dict(right_extra.pop("vlaquantshift"))
        left_protocol.pop("eval_shift")
        right_protocol.pop("eval_shift")
        if left_protocol != right_protocol:
            raise ValueError(f"{label}: calibration or other protocol metadata changed between evaluation conditions")
        if left_extra != right_extra:
            raise ValueError(f"{label}: quantization/calibration or execution metadata changed between conditions")
    return runs


def _interval(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"point": float(point), "low": float(low), "high": float(high), "confidence_level": 0.95}


def _paired_outcomes(fp: np.ndarray, quantized: np.ndarray) -> dict[str, int]:
    return {
        "both_success": int(np.sum((fp == 1) & (quantized == 1))),
        "both_failure": int(np.sum((fp == 0) & (quantized == 0))),
        "quantization_failures": int(np.sum((fp == 1) & (quantized == 0))),
        "quantization_rescues": int(np.sum((fp == 0) & (quantized == 1))),
    }


def analyze_factorial(
    fp_clean, q_clean, fp_shift, q_shift, *, n_bootstrap: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Analyze exactly paired FP/quantized × clean/shift outcomes.

    Primary estimand: equal-weight mean across tasks of
    ``(FP_shift - Q_shift) - (FP_clean - Q_clean)``. Positive means the
    quantization penalty is larger under shift. The percentile bootstrap samples
    tasks, keeping every episode and all four arms of each task together. Its
    uncertainty concerns task sampling, conditional on recorded episodes; it does
    not independently model new episode seeds within a task.
    """
    _integer(n_bootstrap, "n_bootstrap", 2)
    _integer(seed, "bootstrap seed")
    runs = _matched_runs((fp_clean, q_clean, fp_shift, q_shift))
    keys = sorted(runs[0].episodes)
    outcomes = np.array([[run.episodes[key]["success"] for run in runs] for key in keys], dtype=np.float64)
    task_indices = defaultdict(list)
    for index, key in enumerate(keys):
        task_indices[key[:2]].append(index)
    task_keys = sorted(task_indices)
    task_rates = np.array([outcomes[task_indices[key]].mean(axis=0) for key in task_keys])
    task_clean = task_rates[:, 0] - task_rates[:, 1]
    task_shift = task_rates[:, 2] - task_rates[:, 3]
    task_excess = task_shift - task_clean
    rng = np.random.default_rng(seed)
    # Batching avoids allocating n_bootstrap × n_tasks × n_arms for large runs.
    bootstrap_rates = np.empty((n_bootstrap, 4), dtype=np.float64)
    for start in range(0, n_bootstrap, 512):
        stop = min(start + 512, n_bootstrap)
        draw = rng.integers(0, len(task_keys), size=(stop - start, len(task_keys)))
        bootstrap_rates[start:stop] = task_rates[draw].mean(axis=1)
    bootstrap_clean = bootstrap_rates[:, 0] - bootstrap_rates[:, 1]
    bootstrap_shift = bootstrap_rates[:, 2] - bootstrap_rates[:, 3]
    bootstrap_excess = bootstrap_shift - bootstrap_clean
    task_effects = []
    for index, task in enumerate(task_keys):
        members = task_indices[task]
        task_effects.append({
            "suite": task[0], "task_id": task[1],
            "task_name": runs[0].episodes[keys[members[0]]]["task_name"],
            "n_episode_pairs": len(members),
            "success_rates": dict(zip(ARM_NAMES, map(float, task_rates[index]))),
            "quantization_penalty_clean": float(task_clean[index]),
            "quantization_penalty_shift": float(task_shift[index]),
            "excess_degradation": float(task_excess[index]),
        })
    per_episode_excess = (outcomes[:, 2] - outcomes[:, 3]) - (outcomes[:, 0] - outcomes[:, 1])
    single_task = len(task_keys) == 1
    return {
        "schema_version": 1,
        "analysis": "paired_quantization_by_observation_shift",
        "estimand": "task_balanced_excess_quantization_degradation",
        "formula": "(success_fp_shift - success_q_shift) - (success_fp_clean - success_q_clean)",
        "positive_means": "quantization penalty worsens under observation shift",
        "units": "success_probability_difference",
        "protocol": {
            "eval_shift": _protocol(runs[2].header)["eval_shift"],
            "calib_shift": _protocol(runs[1].header)["calib_shift"],
            "shift_seed": _protocol(runs[0].header)["shift_seed"],
            "calibration_metadata_matched": True,
            "calibration_cache_sha256": _calibration_hash(runs[1].header),
            "checkpoint_identity_limit": (
                "Checkpoint path/reference and recorded metadata match; checkpoint content digests are not verified."
            ),
            "pairing": ["suite", "task_id", "episode", "seed"],
            "task_coverage": (
                "explicit_task_list_verified" if (
                    _protocol(runs[0].header)["tasks"] is not None
                    or "episodes_by_task" in _protocol(runs[0].header)
                )
                else "observed_tasks_only; full_suite_completeness_cannot_be_verified_offline"
            ),
        },
        "arms": {
            name: {
                "path": run.path,
                "n_episodes": len(keys),
                "successes": int(outcomes[:, index].sum()),
                "pooled_success_rate": float(outcomes[:, index].mean()),
                "task_balanced_success_rate": _interval(task_rates[:, index].mean(), bootstrap_rates[:, index]),
            }
            for index, (name, run) in enumerate(zip(ARM_NAMES, runs))
        },
        "quantization_penalty": {
            "clean": _interval(task_clean.mean(), bootstrap_clean),
            "shift": _interval(task_shift.mean(), bootstrap_shift),
        },
        "excess_degradation": _interval(task_excess.mean(), bootstrap_excess),
        "pooled_excess_degradation": float(per_episode_excess.mean()),
        "task_effects": task_effects,
        "paired_outcomes": {
            "clean": _paired_outcomes(outcomes[:, 0], outcomes[:, 1]),
            "shift": _paired_outcomes(outcomes[:, 2], outcomes[:, 3]),
            "positive_excess_episode_count": int(np.sum(per_episode_excess > 0)),
            "negative_excess_episode_count": int(np.sum(per_episode_excess < 0)),
            "zero_excess_episode_count": int(np.sum(per_episode_excess == 0)),
            "four_arm_patterns": dict(sorted(Counter("".join(str(int(x)) for x in row) for row in outcomes).items())),
            "pattern_arm_order": list(ARM_NAMES),
        },
        "sample_sizes": {"n_tasks": len(task_keys), "n_episode_pairs": len(keys), "n_rollouts": 4 * len(keys)},
        "bootstrap": {
            "method": "paired_task_cluster_percentile",
            "n_resamples": n_bootstrap,
            "seed": seed,
            "confidence_level": 0.95,
            "estimand_weighting": "equal task weights; all episodes retained within sampled tasks",
            "conditional_on": "recorded episodes within each task",
            "single_task_degenerate": single_task,
            "zero_empirical_task_effect_variance": bool(np.ptp(task_excess) == 0),
            "limitation": (
                "One task cannot estimate between-task uncertainty; the collapsed interval is not evidence of certainty."
                if single_task else None
            ),
        },
    }


def analyze_cross_replay(
    fp_on_fp, q_on_fp, fp_on_q, q_on_q, *, scale=None,
) -> dict[str, Any]:
    """Decompose actions from a 2×2 policy-by-recorded-trajectory replay.

    Each array is T×D with aligned time points, action dimensions and replay
    state/reset convention. The caller is responsible for this alignment.
    ``scale`` is a strictly positive scalar or D-vector and divides action
    dimensions before analysis. This is an algebraic diagnostic, not causal
    mediation: direct + drift + interaction = total as vectors. Norms do not add.
    """
    arrays = []
    names = ("fp_on_fp", "q_on_fp", "fp_on_q", "q_on_q")
    for name, value in zip(names, (fp_on_fp, q_on_fp, fp_on_q, q_on_q)):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite numeric T×D array") from exc
        if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite nonempty T×D array")
        if arrays and array.shape != arrays[0].shape:
            raise ValueError("all four replay arrays must have exactly the same T×D shape")
        arrays.append(array)
    normalizer = np.asarray(1.0 if scale is None else scale, dtype=np.float64)
    if normalizer.shape not in ((), (arrays[0].shape[1],)):
        raise ValueError("scale must be a positive scalar or one value per action dimension")
    if not np.isfinite(normalizer).all() or np.any(normalizer <= 0):
        raise ValueError("scale must contain only positive finite values")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fp_f, q_f, fp_q, q_q = [array / normalizer for array in arrays]
        direct = q_f - fp_f
        drift = fp_q - fp_f
        interaction = q_q - q_f - fp_q + fp_f
        total = q_q - fp_f
    terms = {"direct": direct, "drift": drift, "interaction": interaction, "total": total}
    if any(not np.isfinite(term).all() for term in terms.values()):
        raise ValueError("normalization/decomposition overflowed; use a meaningful positive action scale")
    with np.errstate(over="ignore", invalid="ignore"):
        squared_norms = {name: float(np.sum(term * term)) for name, term in terms.items()}
        cross_terms = {
            "twice_direct_dot_drift": float(2 * np.sum(direct * drift)),
            "twice_direct_dot_interaction": float(2 * np.sum(direct * interaction)),
            "twice_drift_dot_interaction": float(2 * np.sum(drift * interaction)),
        }
    if not all(math.isfinite(value) for value in (*squared_norms.values(), *cross_terms.values())):
        raise ValueError("squared action norms overflowed; use a meaningful positive action scale")
    reconstructed_squared = sum(squared_norms[name] for name in ("direct", "drift", "interaction")) + sum(cross_terms.values())
    report = {
        "schema_version": 1,
        "analysis": "algebraic_cross_replay_action_decomposition",
        "shape": list(arrays[0].shape),
        "scale": normalizer.tolist(),
        "formula": {
            "direct": "q_on_fp - fp_on_fp",
            "drift": "fp_on_q - fp_on_fp",
            "interaction": "q_on_q - q_on_fp - fp_on_q + fp_on_fp",
            "total": "q_on_q - fp_on_fp",
        },
        "vectors": {name: value.tolist() for name, value in terms.items()},
        "frobenius_norms": {name: math.sqrt(value) for name, value in squared_norms.items()},
        "squared_frobenius_norms": squared_norms,
        "signed_cross_terms": cross_terms,
        "squared_norm_identity": {
            "formula": "||total||² = ||direct||² + ||drift||² + ||interaction||² + 2<direct,drift> + 2<direct,interaction> + 2<drift,interaction>",
            "reconstructed_total_squared": reconstructed_squared,
            "absolute_residual": abs(squared_norms["total"] - reconstructed_squared),
        },
        "vector_identity_max_absolute_residual": float(np.max(np.abs(total - direct - drift - interaction))),
        "interpretation": "Algebraic replay diagnostic; not causal mediation. Vector components add; their norms do not.",
        "alignment_requirement": "Caller must align replay time points, action dimensions, history and policy reset conventions.",
    }
    _finite_json(report, "cross-replay report")
    return report
