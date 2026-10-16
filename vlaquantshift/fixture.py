"""Small synthetic controller to exercise the real VQB quantizer and rollout loop.

This is a software integration fixture, NOT a VLA model or a robotics benchmark.
The RGB image encodes a 2D point. A two-layer linear controller tracks a target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from vlaquantbench.benchmarks.base import BenchmarkRunner, EpisodeOutcome
from vlaquantbench.components import ComponentMap
from vlaquantbench.models.base import Observation, TaskSpec, VLAAdapter


class FixtureController(VLAAdapter):
    name = "fixture_controller"
    default_dtype = "fp32"
    supported_benchmarks = ("fixture_reach",)

    def load(self):
        # Exact linear feedback in FP, factorized through a fixed random basis.
        # Never optimize the fixture weights against reported quantized outcomes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(173)
            self.model = nn.Sequential(nn.Linear(8, 32, bias=False), nn.Linear(32, 2, bias=False))
            first = torch.randn(32, 8) * 0.2
            target = torch.zeros(2, 8)
            target[0, 0], target[0, 3] = -0.45, 0.45
            target[1, 1], target[1, 4] = -0.45, 0.45
            with torch.no_grad():
                self.model[0].weight.copy_(first)
                self.model[1].weight.copy_(target @ torch.linalg.pinv(first))
        self.model.to(device=self.device, dtype=self.dtype).eval()

    def build_component_map(self):
        cmap = ComponentMap()
        cmap.add("ve", "image_projection", self.model[0])
        cmap.add("ah", "action_projection", self.model[1])
        return cmap

    def reset(self, task):
        self.target = np.asarray(task.extra["target"], dtype=np.float32)

    def act(self, obs, task):
        rgb = obs.images["front"].mean(axis=(0, 1)).astype(np.float32) / 127.5 - 1.0
        features = np.concatenate([rgb, self.target, [0.13, -0.21, 1.0]])
        x = torch.as_tensor(features, device=self.device, dtype=self.dtype)
        with torch.inference_mode(), self.inference_meter.measure():
            action = self.model(x).float().cpu().numpy()
        return action


class FixtureRunner(BenchmarkRunner):
    name = "fixture_reach"

    @classmethod
    def suites(cls):
        return ("synthetic_v1",)

    @classmethod
    def default_episodes_per_task(cls):
        return 8

    def tasks(self):
        targets = [(-0.6, -0.5), (-0.5, 0.4), (0.5, -0.4), (0.6, 0.5)]
        return [TaskSpec(self.name, self.suite, i, f"synthetic_target_{i}", f"Move to target {i}",
                         self.max_steps(24), {"target": target}) for i, target in enumerate(targets)]

    def run_episode(self, adapter, task, episode, meter):
        adapter.reset(task)
        rng = np.random.default_rng(self.episode_seed(task, episode))
        position = rng.uniform(-0.9, 0.9, 2)
        initial = position.copy()
        target = np.array(task.extra["target"])
        success = False
        for step in range(task.max_steps):
            rgb = np.array([position[0], position[1], 0.17])
            image = np.broadcast_to(np.rint((np.clip(rgb, -1, 1) + 1) * 127.5).astype(np.uint8), (12, 12, 3)).copy()
            observation = Observation({"front": image}, task.instruction, position.copy(), step)
            with meter.measure():
                action = adapter.act(observation, task)
            position = np.clip(position + np.clip(action, -0.25, 0.25), -1, 1)
            distance = float(np.linalg.norm(position - target))
            if distance < 0.035:
                success = True
                break
        return EpisodeOutcome(success, step + 1, extra={"synthetic": True, "initial_position": initial.tolist(),
                                                        "final_distance": distance, "max_steps": task.max_steps})


def register_fixture():
    from vlaquantbench.registry import BENCHMARKS, MODELS
    MODELS["fixture_controller"] = "vlaquantshift.fixture:FixtureController"
    BENCHMARKS["fixture_reach"] = "vlaquantshift.fixture:FixtureRunner"


def run_demo(output, *, device="cpu", episodes=8, n_bootstrap=500):
    import json
    from vlaquantbench.cli import main as vqb
    from .analysis import analyze_factorial
    from .protocol import build_plan

    register_fixture()
    config = {"schema_version": 1, "model": "fixture_controller", "checkpoint": "builtin-fixed-v1",
              "benchmark": "fixture_reach", "suite": "synthetic_v1", "dtype": "fp32", "device": device,
              "tasks": [0, 1, 2, 3], "seeds": [0], "shift_seed": 19,
              "episodes": {"start": 0, "stop": episodes},
              "conditions": ["clean", "brightness:0.2", "gaussian_noise:0.05"],
              "quantizations": [{"name": "rtn_w4a8", "preset": "W4A8", "scope": "e2e"},
                                {"name": "rtn_w3", "preset": "W3", "scope": "e2e"}]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(config, str(output / "runs"))
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for job in plan["jobs"]:
        vqb(job["argv"][3:])
    reports = []
    for comparison in plan["comparisons"]:
        paths = [comparison[arm] for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift")]
        reports.append({"comparison": comparison, "analysis": analyze_factorial(*paths, n_bootstrap=n_bootstrap)})
    report = {"evidence_type": "synthetic_integration_fixture", "robotics_claim": False,
              "description": "Fixed linear image/goal controller; real VQB RTN quantization and closed-loop runner. Not a VLA.",
              "plan_sha256": plan["plan_sha256"], "comparisons": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    from vlaquantbench.cli import main
    register_fixture()
    main()
