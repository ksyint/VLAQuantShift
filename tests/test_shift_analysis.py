import copy
import json

import numpy as np
import pytest

from vlaquantshift.analysis import analyze_cross_replay, analyze_factorial, load_run


def _records(arm, outcomes=None, *, unequal=False):
    if outcomes is None:
        outcomes = {
            "fp_clean": [[True, True], [True, True]],
            "q_clean": [[True, True], [True, False]],
            "fp_shift": [[True, True], [True, False]],
            "q_shift": [[True, False], [False, False]],
        }[arm]
    header = {
        "kind": "header", "model": "toy", "checkpoint": "toy@revision-1",
        "benchmark": "libero", "suite": "libero_spatial",
        "preset": "BASELINE" if arm.startswith("fp_") else "W4",
        "scope": "e2e", "method": "rtn", "baseline_dtype": "bfloat16",
        "episodes_per_task": 2, "seed": 0, "quantize_lm_head": False,
        "include_conv": False,
        "quant_report": {"llm": {"layers": 0 if arm.startswith("fp_") else 1}},
        "extra": {"adapter_options": {}, "runner_options": {}, "vlaquantshift": {
            "schema_version": 1,
            "eval_shift": "clean" if arm.endswith("clean") else "brightness:0.2",
            "calib_shift": "clean", "shift_seed": 7,
            "episodes_from": 0, "episodes_to": 2, "tasks": "0,1", "max_steps": None,
        }},
    }
    episodes = []
    for task_id, task_outcomes in enumerate(outcomes):
        for episode, success in enumerate(task_outcomes):
            episodes.append({
                "kind": "episode", "suite": "libero_spatial", "task_id": task_id,
                "task_name": f"task_{task_id}", "episode": episode,
                "seed": task_id * 1009 + episode, "success": success, "steps": 10, "extra": {},
            })
    if unequal:
        header["extra"]["vlaquantshift"]["episodes_by_task"] = {
            str(task_id): list(range(len(values))) for task_id, values in enumerate(outcomes)
        }
    return [header, *episodes]


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _four_runs(tmp_path):
    return [_write(tmp_path / f"{arm}.jsonl", _records(arm)) for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift")]


def test_known_factorial_effect_and_paired_failure_rescue_counts(tmp_path):
    runs = _four_runs(tmp_path)
    report = analyze_factorial(*runs, n_bootstrap=500, seed=19)
    assert report["estimand"] == "task_balanced_excess_quantization_degradation"
    assert report["quantization_penalty"]["clean"]["point"] == 0.25
    assert report["quantization_penalty"]["shift"]["point"] == 0.5
    assert report["excess_degradation"] == {"point": 0.25, "low": 0.0, "high": 0.5, "confidence_level": 0.95}
    assert [task["excess_degradation"] for task in report["task_effects"]] == [0.5, 0.0]
    assert report["paired_outcomes"]["clean"]["quantization_failures"] == 1
    assert report["paired_outcomes"]["shift"]["quantization_failures"] == 2
    assert report["sample_sizes"] == {"n_tasks": 2, "n_episode_pairs": 4, "n_rollouts": 16}
    assert report == analyze_factorial(*runs, n_bootstrap=500, seed=19)
    json.dumps(report, allow_nan=False)


def test_unequal_task_sizes_use_task_balanced_estimand(tmp_path):
    conditions = {
        "fp_clean": [[True], [True, True]],
        "q_clean": [[True], [True, True]],
        "fp_shift": [[True], [True, True]],
        "q_shift": [[False], [True, True]],
    }
    paths = [_write(tmp_path / f"{arm}.jsonl", _records(arm, values, unequal=True)) for arm, values in conditions.items()]
    report = analyze_factorial(*paths, n_bootstrap=100, seed=2)
    assert report["excess_degradation"]["point"] == 0.5
    assert report["pooled_excess_degradation"] == pytest.approx(1 / 3)
    assert report["bootstrap"]["method"] == "paired_task_cluster_percentile"
    assert [task["n_episode_pairs"] for task in report["task_effects"]] == [1, 2]


def test_single_task_interval_flag_and_quantization_rescue(tmp_path):
    paths = []
    for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift"):
        records = _records(arm, [[not arm.startswith("fp_"), False]])
        records[0]["extra"]["vlaquantshift"]["tasks"] = "0"
        paths.append(_write(tmp_path / f"{arm}.jsonl", records))
    report = analyze_factorial(*paths, n_bootstrap=20)
    assert report["bootstrap"]["single_task_degenerate"] is True
    assert "cannot estimate" in report["bootstrap"]["limitation"]
    assert report["excess_degradation"]["low"] == report["excess_degradation"]["high"] == 0.0
    assert report["paired_outcomes"]["clean"]["quantization_rescues"] == 1


@pytest.mark.parametrize("mutation, match", [
    (lambda records: records.append(copy.deepcopy(records[1])), "duplicate episode key"),
    (lambda records: records.append(copy.deepcopy(records[0])), "kind='episode'"),
    (lambda records: records.pop(), "episode coverage"),
    (lambda records: records.__delitem__(slice(3, 5)), "task coverage"),
    (lambda records: records[1].update(success=1), "must be a bool"),
    (lambda records: records[1].update(success="false"), "must be a bool"),
    (lambda records: records[1].pop("success"), "must be a bool"),
    (lambda records: records[1].update(suite="different"), "suite differs"),
    (lambda records: records[2].update(task_name="wrong"), "inconsistent task_name"),
    (lambda records: records[1].update(wall_time_s=float("nan")), "nonfinite"),
    (lambda records: records[0]["extra"].pop("vlaquantshift"), "missing extra.vlaquantshift"),
    (lambda records: records[0]["extra"]["vlaquantshift"].update(schema_version=True), "schema_version"),
    (lambda records: records[0]["extra"]["vlaquantshift"].update(episodes_from=2), "episode interval"),
    (lambda records: records[0]["extra"]["vlaquantshift"].update(tasks="0,0"), "duplicate task IDs"),
    (lambda records: records[0]["extra"]["vlaquantshift"].update(eval_shift="brightness:0.20"), "canonical"),
])
def test_strict_run_rejections(tmp_path, mutation, match):
    records = _records("fp_clean")
    mutation(records)
    with pytest.raises(ValueError, match=match):
        load_run(_write(tmp_path / "broken.jsonl", records))


@pytest.mark.parametrize("line", ['{"kind":', "\x00", "", 'null', '{"kind":"episode","kind":"episode"}'])
def test_corrupt_lines_are_never_silently_dropped(tmp_path, line):
    path = _write(tmp_path / "broken.jsonl", _records("fp_clean"))
    with path.open("a") as handle:
        handle.write(line + "\n")
    with pytest.raises(ValueError):
        load_run(path)


def test_empty_file_is_invalid(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    with pytest.raises(ValueError, match="missing header"):
        load_run(path)


def test_same_episode_with_changed_seed_is_duplicate_position(tmp_path):
    records = _records("fp_clean")
    other = copy.deepcopy(records[1])
    other["seed"] += 99
    records.append(other)
    with pytest.raises(ValueError, match="conflicting seed"):
        load_run(_write(tmp_path / "broken.jsonl", records))


@pytest.mark.parametrize("field, value", [
    ("checkpoint", "new-revision"), ("model", "other"), ("suite", "different"),
    ("benchmark", "calvin"), ("seed", 9), ("baseline_dtype", "float32"),
    ("preset", "W8"), ("method", "awq"), ("scope", "llm"),
    ("include_conv", True), ("quantize_lm_head", True), ("quant_report", {"w_bits": 3}),
])
def test_factorial_rejects_changed_model_or_quantization_config(tmp_path, field, value):
    paths = _four_runs(tmp_path)
    records = _records("q_shift")
    records[0][field] = value
    _write(paths[3], records)
    with pytest.raises(ValueError):
        analyze_factorial(*paths, n_bootstrap=20)


@pytest.mark.parametrize("field, value", [
    ("calib_shift", "gaussian_noise:0.1"), ("shift_seed", 91),
    ("eval_shift", "occlusion:0.1"), ("max_steps", 200),
])
def test_factorial_rejects_changed_shift_and_calibration_metadata(tmp_path, field, value):
    paths = _four_runs(tmp_path)
    records = _records("q_shift")
    records[0]["extra"]["vlaquantshift"][field] = value
    _write(paths[3], records)
    with pytest.raises(ValueError):
        analyze_factorial(*paths, n_bootstrap=20)


def test_changed_calibration_path_is_rejected(tmp_path):
    paths = _four_runs(tmp_path)
    records = _records("q_shift")
    records[0]["extra"]["calib"] = "different_samples.jsonl"
    _write(paths[3], records)
    with pytest.raises(ValueError, match="calibration"):
        analyze_factorial(*paths, n_bootstrap=20)


@pytest.mark.parametrize("report", [{}, {"llm": {"layers": 0}}, {"layers": 0}, {"llm": {"layers": True}}])
def test_quantized_arms_require_positive_applied_layer_count(tmp_path, report):
    paths = _four_runs(tmp_path)
    for index, arm in ((1, "q_clean"), (3, "q_shift")):
        records = _records(arm)
        records[0]["quant_report"] = report
        _write(paths[index], records)
    with pytest.raises(ValueError, match="layer"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_baseline_cannot_report_applied_quantized_layers(tmp_path):
    paths = _four_runs(tmp_path)
    records = _records("fp_clean")
    records[0]["quant_report"]["llm"]["layers"] = 1
    _write(paths[0], records)
    with pytest.raises(ValueError, match="claims BASELINE"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_method_level_applied_layer_report_is_supported(tmp_path):
    paths = _four_runs(tmp_path)
    for index, arm in ((1, "q_clean"), (3, "q_shift")):
        records = _records(arm)
        records[0]["quant_report"] = {"layers": 1, "w_bits": 4}
        _write(paths[index], records)
    assert analyze_factorial(*paths, n_bootstrap=20)["excess_degradation"]["point"] == 0.25


@pytest.mark.parametrize("hashes, match", [
    ((None, None), "no calibration cache_sha256"),
    (("a" * 64, "b" * 64), "mismatched calibration cache_sha256"),
    (("not-a-hash", "not-a-hash"), "hexadecimal digest"),
])
def test_calibration_requires_matching_content_hash(tmp_path, hashes, match):
    paths = _four_runs(tmp_path)
    for index, arm, digest in ((1, "q_clean", hashes[0]), (3, "q_shift", hashes[1])):
        records = _records(arm)
        records[0]["extra"]["act_calib"] = ["llm"]
        records[0]["extra"]["act_calib_config"] = {"episodes": 2, "cache": "calibration.pt"}
        if digest is not None:
            records[0]["extra"]["act_calib_config"]["cache_sha256"] = digest
        _write(paths[index], records)
    with pytest.raises(ValueError, match=match):
        analyze_factorial(*paths, n_bootstrap=20)


def test_matching_calibration_hash_is_reported_without_checkpoint_digest_claim(tmp_path):
    paths = _four_runs(tmp_path)
    for index, arm in ((1, "q_clean"), (3, "q_shift")):
        records = _records(arm)
        records[0]["extra"]["act_calib"] = ["llm"]
        records[0]["extra"]["act_calib_config"] = {"cache_sha256": "a" * 64}
        _write(paths[index], records)
    protocol = analyze_factorial(*paths, n_bootstrap=20)["protocol"]
    assert protocol["calibration_metadata_matched"] is True
    assert protocol["calibration_cache_sha256"] == "a" * 64
    assert "digests are not verified" in protocol["checkpoint_identity_limit"]
    assert "calibration_frozen_across_eval_conditions" not in protocol


def test_pairing_uses_actual_episode_seed_no_inner_join(tmp_path):
    paths = _four_runs(tmp_path)
    records = _records("q_shift")
    records[1]["seed"] += 33
    _write(paths[3], records)
    with pytest.raises(ValueError, match="exact episode/seed pairing"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_missing_identical_episode_in_all_arms_still_rejected(tmp_path):
    paths = []
    for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift"):
        paths.append(_write(tmp_path / f"{arm}.jsonl", _records(arm)[:-1]))
    with pytest.raises(ValueError, match="episode coverage"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_fp_alias_cannot_masquerade_as_quantized_arm(tmp_path):
    paths = _four_runs(tmp_path)
    for index, arm in ((1, "q_clean"), (3, "q_shift")):
        records = _records(arm)
        records[0]["preset"] = "BF16"
        _write(paths[index], records)
    with pytest.raises(ValueError, match="non-BASELINE"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_episode_audit_metadata_must_agree_with_header(tmp_path):
    records = _records("fp_clean")
    records[1]["extra"]["vlaquantshift"] = {
        "schema_version": 1, "eval_shift": "clean", "shift_seed": 7, "policy_seed": 999,
    }
    with pytest.raises(ValueError, match="policy_seed"):
        load_run(_write(tmp_path / "broken.jsonl", records))


def test_episode_audit_and_extended_protocol_metadata_are_supported(tmp_path):
    paths = []
    for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift"):
        records = _records(arm)
        protocol = records[0]["extra"]["vlaquantshift"]
        protocol.update(episode_rng="identity_seed_v1", protocol_version=1, code_revision="abc1234")
        for episode in records[1:]:
            episode["extra"]["vlaquantshift"] = {
                "schema_version": 1, "eval_shift": protocol["eval_shift"],
                "shift_seed": 7, "policy_seed": episode["seed"], "n_actions": 10,
            }
        paths.append(_write(tmp_path / f"{arm}.jsonl", records))
    assert analyze_factorial(*paths, n_bootstrap=20)["excess_degradation"]["point"] == 0.25


@pytest.mark.parametrize("where", ["header", "protocol"])
def test_code_revision_cannot_change_between_arms(tmp_path, where):
    paths = _four_runs(tmp_path)
    records = _records("q_shift")
    target = records[0] if where == "header" else records[0]["extra"]["vlaquantshift"]
    target["code_revision"] = "changed"
    _write(paths[3], records)
    with pytest.raises(ValueError, match="code_revision"):
        analyze_factorial(*paths, n_bootstrap=20)


def test_actual_planned_task_episode_coverage_is_enforced(tmp_path):
    records = _records("fp_clean")
    records[0]["extra"]["vlaquantshift"]["episodes_by_task"] = {"0": [0], "1": [0, 1]}
    with pytest.raises(ValueError, match="episode coverage"):
        load_run(_write(tmp_path / "broken.jsonl", records))


@pytest.mark.parametrize("n_bootstrap", [0, 1, True, 2.5])
def test_bootstrap_requires_at_least_two_replicates(tmp_path, n_bootstrap):
    with pytest.raises(ValueError, match="n_bootstrap"):
        analyze_factorial(*_four_runs(tmp_path), n_bootstrap=n_bootstrap)


def test_cross_replay_exact_decomposition_and_signed_cross_terms():
    baseline = np.array([[0.0, 1.0], [2.0, 3.0]])
    direct = np.array([[1.0, 0.0], [0.0, 2.0]])
    drift = -direct
    interaction = np.ones((2, 2))
    report = analyze_cross_replay(
        baseline, baseline + direct, baseline + drift,
        baseline + direct + drift + interaction,
    )
    for name, expected in (("direct", direct), ("drift", drift), ("interaction", interaction), ("total", interaction)):
        np.testing.assert_allclose(report["vectors"][name], expected)
    assert report["signed_cross_terms"]["twice_direct_dot_drift"] == -10.0
    assert report["squared_norm_identity"]["reconstructed_total_squared"] == 4.0
    assert report["squared_norm_identity"]["absolute_residual"] == 0.0
    assert report["vector_identity_max_absolute_residual"] == 0.0
    assert report["frobenius_norms"]["total"] != sum(report["frobenius_norms"][name] for name in ("direct", "drift", "interaction"))
    json.dumps(report, allow_nan=False)


def test_cross_replay_action_dimension_normalization():
    zero = np.zeros((3, 2))
    q = np.tile([2.0, 3.0], (3, 1))
    report = analyze_cross_replay(zero, q, zero, q, scale=[2.0, 3.0])
    assert report["squared_frobenius_norms"]["direct"] == 6.0
    np.testing.assert_array_equal(report["vectors"]["direct"], np.ones((3, 2)))


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((0, 2)), np.ones((2, 3)), [[float("nan"), 1.0]], [[float("inf"), 1.0]]])
def test_cross_replay_rejects_invalid_shape_or_nonfinite_actions(bad):
    with pytest.raises(ValueError):
        analyze_cross_replay(np.zeros((1, 2)), bad, np.zeros((1, 2)), np.zeros((1, 2)))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), [1.0], [1.0, 0.0], [[1.0, 2.0]]])
def test_cross_replay_rejects_invalid_scales(scale):
    zero = np.zeros((2, 2))
    with pytest.raises(ValueError, match="scale"):
        analyze_cross_replay(zero, zero, zero, zero, scale=scale)


def test_cross_replay_rejects_numeric_overflow():
    zero = np.zeros((2, 2))
    with pytest.raises(ValueError, match="overflowed"):
        analyze_cross_replay(zero, np.full((2, 2), 1e200), zero, zero)
