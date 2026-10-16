"""Auditable wrapper around the original VLAQuantBench adapter interface."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess

import numpy as np
import torch

from .shifts import ShiftSpec, transform_observation


def runtime_provenance(repository):
    """Record installed numerical/simulator versions without importing their stacks."""
    packages = {}
    for name in ("torch", "numpy", "mujoco", "mujoco-mjx", "dm-control", "robosuite", "libero",
                 "transformers", "lerobot", "VLABench", "sapien", "mani-skill2"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    code = {"commit": None, "dirty": None}
    try:
        code["commit"] = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        code["dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"python": platform.python_version(), "packages": packages, "code": code}


def calibration_cache_path(path, identity: dict):
    """Separate caches for shift, seed, adapter, runner, and calibration protocol."""
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return path.with_name(f"{path.stem}-shift-{tag}{path.suffix}")


def flatten_action(action) -> np.ndarray:
    # VLABench and CALVIN return ragged tuples (position, rotation, gripper).
    if isinstance(action, (tuple, list)):
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in action])
    return np.asarray(action, dtype=np.float64).reshape(-1)


class ObservationIntervention:
    def __init__(self, adapter, shift: ShiftSpec | str, seed: int = 0):
        self.adapter = adapter
        self.shift = ShiftSpec.parse(shift)
        self.shift_seed = int(seed)
        self._episode_key = None
        self._policy_seed = None
        self._observations = hashlib.sha256()
        self._norms = []
        self._variation = []
        self._previous = None

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def begin_episode(self, task, episode: int, seed: int):
        self._episode_key = (task.benchmark, task.suite, task.task_id, int(episode))
        self._policy_seed = int(seed)
        self._observations = hashlib.sha256()
        self._norms = []
        self._variation = []
        self._previous = None
        hook = getattr(self.adapter, "begin_episode", None)
        if hook is not None:
            hook(task, episode, seed)

    def reset(self, task):
        if self._episode_key is None:
            raise RuntimeError("begin_episode must be called before reset")
        self.adapter.reset(task)
        # Upstream stochastic adapters increment a private seed in reset.
        # Override it AFTER reset so cache hits, task order, and resume do not
        # change the actual inference noise of a given episode.
        if hasattr(self.adapter, "_episode_seed"):
            self.adapter._episode_seed = self._policy_seed
        random.seed(self._policy_seed)
        np.random.seed(self._policy_seed % (2**32))
        torch.manual_seed(self._policy_seed)

    def act(self, obs, task):
        if self._episode_key is None:
            raise RuntimeError("begin_episode must be called before act")
        # Include the total call index: CALVIN resets obs.step at each subtask.
        shifted = transform_observation(obs, self.shift, seed=self.shift_seed,
                                        episode_key=(*self._episode_key, len(self._norms)))
        self._observations.update(str(int(obs.step)).encode())
        self._observations.update(shifted.instruction.encode())
        for camera, pixels in sorted(shifted.images.items()):
            self._observations.update(camera.encode())
            self._observations.update(str(pixels.shape).encode())
            self._observations.update(pixels.tobytes())
        if shifted.state is not None:
            self._observations.update(np.asarray(shifted.state).tobytes())
        action = self.adapter.act(shifted, task)
        values = flatten_action(action)
        if not np.all(np.isfinite(values)):
            raise ValueError("policy returned non-finite action")
        self._norms.append(float(np.linalg.norm(values)))
        if self._previous is not None:
            if values.shape != self._previous.shape:
                raise ValueError("action dimensions changed within episode")
            self._variation.append(float(np.linalg.norm(values - self._previous)))
        self._previous = values.copy()
        return action

    def episode_summary(self):
        return {
            "schema_version": 1, "eval_shift": self.shift.canonical,
            "shift_seed": self.shift_seed, "policy_seed": self._policy_seed,
            "observations_sha256": self._observations.hexdigest(), "n_actions": len(self._norms),
            "action_l2_mean": float(np.mean(self._norms)) if self._norms else None,
            "action_temporal_variation_mean": float(np.mean(self._variation)) if self._variation else None,
            "action_units": "native adapter units; descriptive within model/protocol only",
        }
