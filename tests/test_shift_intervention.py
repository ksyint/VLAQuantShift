from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from vlaquantbench.benchmarks.base import BenchmarkRunner, EpisodeOutcome
from vlaquantbench.components import ComponentMap
from vlaquantbench.models.base import Observation, TaskSpec, VLAAdapter
from vlaquantbench.results import RunHeader, RunWriter, read_run
from vlaquantshift.intervention import ObservationIntervention, calibration_cache_path
from vlaquantshift.shifts import ShiftSpec, shift_image, transform_observation


class ToyAdapter(VLAAdapter):
    name = "toy"

    def load(self):
        self.model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[.2, .3, .4], [.1, -.2, .3]]))
        self._episode_seed = 99
        self.seen = []

    def build_component_map(self):
        return ComponentMap(ah=[("head", self.model)])

    def reset(self, task):
        self._episode_seed = self._episode_seed * 7919 % (2**31 - 1)
        torch.manual_seed(self._episode_seed)

    def act(self, obs, task):
        self.seen.append(obs)
        value = torch.as_tensor(obs.images["main"].mean((0, 1)) / 255, dtype=torch.float32)
        return self.model(value).detach().numpy() + torch.rand(2).numpy() * .01


class ToyRunner(BenchmarkRunner):
    name = "calvin"

    @classmethod
    def suites(cls):
        return ("toy",)

    @classmethod
    def default_episodes_per_task(cls):
        return 2

    def tasks(self):
        return [TaskSpec(self.name, self.suite, 0, "toy_task", "move", 3)]

    def run_episode(self, adapter, task, episode, meter):
        adapter.reset(task)
        for step in range(3):
            pixels = np.full((8, 8, 3), 40 + step + episode, dtype=np.uint8)
            obs = Observation({"main": pixels}, "move", np.ones(3), step, {"rgb": pixels})
            with meter.measure():
                adapter.act(obs, task)
        return EpisodeOutcome(success=True, steps=3)


def toy():
    adapter = ToyAdapter("synthetic", device="cpu", dtype="fp32")
    adapter.load()
    return adapter


@pytest.mark.parametrize("condition", ["clean", "brightness:0.2", "gaussian_noise:0.05", "occlusion:0.15", "camera_dropout:0.2"])
def test_shift_is_deterministic_uint8_nonmutating(condition):
    source = np.arange(12 * 15 * 3, dtype=np.uint8).reshape(12, 15, 3)
    before = source.copy()
    np.random.seed(19)
    state = np.random.get_state()
    first = shift_image(source, condition, seed=42)
    second = shift_image(source, condition, seed=42)
    assert np.array_equal(first, second)
    assert np.array_equal(source, before)
    assert first.dtype == np.uint8 and not np.shares_memory(first, source)
    assert np.array_equal(state[1], np.random.get_state()[1])


@pytest.mark.parametrize("condition", ["unknown:0.2", "clean:0", "brightness:nan", "occlusion:-0.1", "camera_dropout:1.1"])
def test_invalid_shift_rejected(condition):
    with pytest.raises(ValueError):
        ShiftSpec.parse(condition)


def test_raw_rgb_and_reversed_wrist_are_shifted_without_mutation():
    front = np.full((4, 5, 3), 80, dtype=np.uint8)
    wrist = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    original_raw = {"obs": {"agentview_image": front[::-1, ::-1], "robot0_eye_in_hand_image": wrist[::-1, ::-1]},
                    "rgb": np.stack([front, wrist]), "controller": {"ee_pos": np.ones(3)}}
    obs = Observation({"main": front, "wrist": wrist}, "move", raw=original_raw)
    changed = transform_observation(obs, "brightness:0.2", seed=1, episode_key=("task", 0))
    assert np.array_equal(changed.raw["obs"]["robot0_eye_in_hand_image"], changed.images["wrist"][::-1, ::-1])
    assert np.array_equal(changed.raw["rgb"][0], changed.images["main"])
    assert np.array_equal(original_raw["rgb"][0], front)
    assert not np.array_equal(changed.images["main"], front)


def test_unmapped_rgb_refused():
    obs = Observation({"main": np.zeros((4, 5, 3), dtype=np.uint8)}, "move",
                      raw={"hidden_camera": np.ones((4, 5, 3), dtype=np.uint8)})
    with pytest.raises(ValueError, match="unmapped raw RGB"):
        transform_observation(obs, "brightness:0.2", seed=1, episode_key=(0,))


def test_identical_independent_cameras_keep_their_own_noise():
    first = np.full((8, 8, 3), 100, dtype=np.uint8)
    second = first.copy()
    obs = Observation({"first": first, "second": second}, "move", raw={"one": first, "two": second})
    changed = transform_observation(obs, "gaussian_noise:0.05", seed=2, episode_key=(0,))
    assert np.array_equal(changed.raw["one"], changed.images["first"])
    assert np.array_equal(changed.raw["two"], changed.images["second"])
    assert not np.array_equal(changed.images["first"], changed.images["second"])
    obs.raw = {"ambiguous_copy": first.copy()}
    with pytest.raises(ValueError, match="ambiguous raw RGB"):
        transform_observation(obs, "gaussian_noise:0.05", seed=2, episode_key=(0,))


def test_canonical_severity_preserves_float_identity():
    assert ShiftSpec.parse("brightness:0.20").canonical == "brightness:0.2"
    assert ShiftSpec.parse("occlusion:1.0").canonical == "occlusion:1"
    assert ShiftSpec.parse("brightness:0.1234561").canonical != ShiftSpec.parse("brightness:0.1234562").canonical


def test_native_runner_hook_trace_and_rng_are_episode_identity_based(tmp_path):
    runner = ToyRunner("toy", seed=4)
    wrapped = ObservationIntervention(toy(), "gaussian_noise:0.05", 71)
    path = tmp_path / "run.jsonl"
    with RunWriter(path, RunHeader()) as writer:
        runner.run(wrapped, writer, episodes=[1, 0, 1], log_every=0)
    _, records = read_run(path)
    # Writer correctly skips duplicate identity within a run.
    assert len(records) == 2
    trace = records[0].extra["vlaquantshift"]
    assert trace["n_actions"] == 3 and trace["policy_seed"] == runner.episode_seed(runner.tasks()[0], 1)
    runner.run(wrapped, episodes=[1], log_every=0)
    assert wrapped.episode_summary() == trace


def test_resume_rejects_shift_dtype_or_calibration_change(tmp_path):
    base = RunHeader(model="toy", baseline_dtype="float32", quant_report={"ah": {"layers": 1}},
                     extra={"vlaquantshift": {"eval_shift": "clean", "calib_shift": "clean"},
                            "act_calib_config": {"alpha": .5}})
    path = tmp_path / "run.jsonl"
    RunWriter(path, base).close()
    candidates = [replace(base, baseline_dtype="bfloat16"), replace(base, quant_report={}),
                  replace(base, extra={**base.extra, "vlaquantshift": {"eval_shift": "brightness:0.2"}}),
                  replace(base, extra={**base.extra, "act_calib_config": {"alpha": .75}})]
    for candidate in candidates:
        with pytest.raises(RuntimeError, match="different configuration"):
            RunWriter(path, candidate)


def test_cache_identity_separates_shift_seed_and_runner(tmp_path):
    path = tmp_path / "cache.pt"
    identities = [{"calib_shift": "clean", "shift_seed": 0},
                  {"calib_shift": "brightness:0.2", "shift_seed": 0},
                  {"calib_shift": "brightness:0.2", "shift_seed": 1},
                  {"calib_shift": "brightness:0.2", "shift_seed": 1, "max_steps": 4}]
    assert len({calibration_cache_path(path, identity) for identity in identities}) == 4


def test_cli_runs_real_upstream_quantization_with_shifted_calibration(tmp_path, monkeypatch, capsys):
    import vlaquantbench.cli as cli
    import vlaquantbench.quant.act_calib as calibration

    monkeypatch.setattr(cli, "get_runner_cls", lambda _: ToyRunner)
    monkeypatch.setattr(cli, "build_adapter", lambda _: toy())
    monkeypatch.setattr(calibration, "cache_path", lambda *a, **kw: tmp_path / "calib.pt")
    for index, condition in enumerate(["clean", "brightness:0.2"]):
        out = tmp_path / f"run{index}.jsonl"
        cli.main(["run", "--model", "openvla", "--benchmark", "calvin", "--suite", "toy",
                  "--checkpoint", "synthetic", "--device", "cpu", "--preset", "W4A4", "--scope", "ah",
                  "--episodes", "2", "--eval-shift", condition, "--calib-shift", condition,
                  "--act-calib", "ah", "--act-calib-from", "2", "--act-calib-episodes", "1", "--out", str(out)])
        header, episodes = read_run(out)
        assert header.quant_report["ah"]["calibrated_layers"] == 1
        assert header.extra["vlaquantshift"]["eval_shift"] == condition
        assert header.extra["act_calib_config"]["calib_shift"] == condition
        assert len(header.extra["act_calib_config"]["cache_sha256"]) == 64
        assert len(episodes) == 2 and all(e.extra["vlaquantshift"]["n_actions"] == 3 for e in episodes)
        json.loads(capsys.readouterr().out)
    assert len(list(tmp_path.glob("calib-shift-*.pt"))) == 2


def test_cli_rejects_calibration_leakage_before_loading_model(monkeypatch):
    import vlaquantbench.cli as cli

    monkeypatch.setattr(cli, "get_runner_cls", lambda _: ToyRunner)
    monkeypatch.setattr(cli, "build_adapter", lambda _: pytest.fail("must reject leakage before model load"))
    with pytest.raises(ValueError, match="overlap"):
        cli.main(["run", "--model", "openvla", "--benchmark", "calvin", "--suite", "toy",
                  "--checkpoint", "synthetic", "--episodes", "3", "--eval-shift", "clean",
                  "--act-calib", "ah", "--act-calib-from", "2"])


def test_cli_rejects_silent_quantization_noop(tmp_path, monkeypatch):
    import vlaquantbench.cli as cli

    monkeypatch.setattr(cli, "get_runner_cls", lambda _: ToyRunner)
    monkeypatch.setattr(cli, "build_adapter", lambda _: toy())
    with pytest.raises(ValueError, match="zero layers"):
        cli.main(["run", "--model", "openvla", "--benchmark", "calvin", "--suite", "toy",
                  "--checkpoint", "synthetic", "--episodes", "2", "--eval-shift", "clean",
                  "--preset", "W4", "--exclude", "*", "--out", str(tmp_path / "noop.jsonl")])
    assert not (tmp_path / "noop.jsonl").exists()
