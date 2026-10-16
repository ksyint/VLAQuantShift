import copy
import json

import pytest

from vlaquantshift.protocol import build_plan, execute_plan, verify_plan


@pytest.fixture
def config():
    return {"schema_version": 1, "model": "pi05", "checkpoint": "lerobot/pi05_libero_finetuned_v044",
            "benchmark": "libero", "suite": "libero_spatial", "tasks": [0, 1], "seeds": [0, 1],
            "episodes": {"start": 0, "stop": 4}, "conditions": ["clean", "gaussian_noise:0.05"],
            "quantizations": [{"name": "rtn", "preset": "W4A8", "scope": "e2e"}]}


def test_complete_crossed_plan_and_stable_identity(config):
    plan = build_plan(config)
    assert len(plan["jobs"]) == 8
    assert plan["total_episodes"] == 64
    assert len(plan["comparisons"]) == 2
    assert verify_plan(json.loads(json.dumps(plan))) == plan
    assert build_plan(copy.deepcopy(config)) == plan
    for comparison in plan["comparisons"]:
        assert len({comparison[x] for x in ("fp_clean", "q_clean", "fp_shift", "q_shift")}) == 4


def test_no_calibration_leakage(config):
    config["calibration"] = {"components": ["ah"], "start": 3, "count": 2, "tasks": 1}
    with pytest.raises(ValueError, match="overlap"):
        build_plan(config)
    config["calibration"]["start"] = 4
    plan = build_plan(config)
    quant = [x for x in plan["jobs"] if x["quantization"] == "rtn"]
    assert all("--act-calib" in x["argv"] for x in quant)
    assert all("--act-calib" not in x["argv"] for x in plan["jobs"] if x["quantization"] == "baseline")


@pytest.mark.parametrize("field,value", [("tasks", [0, 0]), ("seeds", []), ("conditions", ["clean", "clean"]),
                                        ("schema_version", 2), ("checkpoint", "auto")])
def test_invalid_configs_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        build_plan(config)


def test_typos_and_duplicate_quantizers_rejected(config):
    config["quantizatoins"] = []
    with pytest.raises(ValueError, match="unknown"):
        build_plan(config)
    del config["quantizatoins"]
    config["quantizations"].append({"name": "renamed", "preset": "W4A8", "scope": "e2e"})
    with pytest.raises(ValueError, match="duplicate"):
        build_plan(config)


def test_manifest_tampering_rejected(config):
    plan = build_plan(config)
    plan["jobs"][0]["argv"] = ["sh", "-c", "unexpected"]
    with pytest.raises(ValueError, match="differs"):
        verify_plan(plan)


def test_weight_only_calibration_rejected_before_launch(config):
    config["calibration"] = {"components": ["ah"], "start": 4}
    config["quantizations"][0]["preset"] = "W4"
    with pytest.raises(ValueError, match="activation-quantized"):
        build_plan(config)


def test_synthetic_plan_is_standalone(config):
    config.update(model="fixture_controller", benchmark="fixture_reach", checkpoint="builtin-fixed-v1",
                  suite="synthetic_v1", dtype="fp32", device="cpu")
    plan = build_plan(config)
    assert plan["jobs"][0]["argv"][2] == "vlaquantshift.fixture"
    assert verify_plan(plan) == plan


def test_execute_is_shell_free_and_gpu_restricted(config, monkeypatch):
    plan = build_plan(config)
    calls = []
    monkeypatch.setattr("vlaquantshift.protocol.subprocess.run", lambda *a, **kw: calls.append((a, kw)))
    execute_plan(plan, gpu_ids="6,7", job_ids=[plan["jobs"][0]["id"]])
    assert len(calls) == 1
    argv, opts = calls[0]
    assert argv[0][1:3] == ["-m", "vlaquantbench.cli"]
    assert opts["env"]["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert opts.get("shell", False) is False
    with pytest.raises(ValueError, match="unknown job"):
        execute_plan(plan, job_ids=["missing"])
