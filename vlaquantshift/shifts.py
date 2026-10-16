"""Deterministic observation interventions, independent of policy RNG state.

These are controlled sensor corruptions, not native VLABench semantic tracks.
Brightness adds a fraction of 255; noise sigma is a fraction of 255;
occlusion is one random rectangle of approximately the requested area;
camera dropout independently blanks a complete camera with that probability.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShiftSpec:
    kind: str = "clean"
    severity: float = 0.0

    def __post_init__(self):
        if self.kind not in {"clean", "brightness", "gaussian_noise", "occlusion", "camera_dropout"}:
            raise ValueError(f"unsupported observation shift: {self.kind!r}")
        if not np.isfinite(self.severity) or not 0 <= self.severity <= 1:
            raise ValueError("shift severity must be finite and between 0 and 1")
        if self.kind == "clean" and self.severity:
            raise ValueError("clean has no severity")

    @classmethod
    def parse(cls, text: str | "ShiftSpec") -> "ShiftSpec":
        if isinstance(text, cls):
            return text
        parts = text.strip().lower().split(":")
        if parts == ["clean"]:
            return cls()
        if len(parts) != 2 or parts[0] == "clean":
            raise ValueError("use clean or a condition such as gaussian_noise:0.05")
        return cls(parts[0], float(parts[1]))

    @property
    def canonical(self) -> str:
        severity = float(self.severity)
        text = str(int(severity)) if severity.is_integer() else repr(severity)
        return "clean" if self.kind == "clean" else f"{self.kind}:{text}"


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def shift_image(image: np.ndarray, spec: ShiftSpec | str, *, seed: int) -> np.ndarray:
    spec = ShiftSpec.parse(spec)
    source = np.asarray(image)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("observation shifts require HxWx3 uint8 RGB images")
    out = source.copy()
    if not out.size or spec.kind == "clean" or spec.severity == 0:
        return out
    rng = np.random.default_rng(seed)
    if spec.kind == "brightness":
        out = np.clip(out.astype(np.float32) + 255 * spec.severity, 0, 255).astype(np.uint8)
    elif spec.kind == "gaussian_noise":
        out = np.clip(out.astype(np.float32) + rng.normal(0, 255 * spec.severity, out.shape), 0, 255).astype(np.uint8)
    elif spec.kind == "occlusion":
        height, width = out.shape[:2]
        h = max(1, min(height, round(height * np.sqrt(spec.severity))))
        w = max(1, min(width, round(width * np.sqrt(spec.severity))))
        y, x = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
        out[y:y+h, x:x+w] = 0
    elif spec.kind == "camera_dropout" and rng.random() < spec.severity:
        out.fill(0)
    return out


def transform_observation(obs, spec: ShiftSpec | str, *, seed: int, episode_key: tuple):
    """Copy images and every raw RGB alias; reject unmapped raw camera arrays.

    Some adapters read raw RGB instead of Observation.images. Arrays of other
    types/shapes and live simulator objects are retained, never mutated. The
    CLI limits shifted runs to audited adapters to avoid hidden render paths.
    """
    spec = ShiftSpec.parse(spec)
    images = {}
    def same_view(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype and left.strides == right.strides
                and left.__array_interface__["data"][0] == right.__array_interface__["data"][0])

    # Duplicate camera aliases (SIMPLER main/overhead) receive the same image.
    originals: list[tuple[np.ndarray, np.ndarray]] = []
    for camera in sorted(obs.images):
        src = np.asarray(obs.images[camera])
        alias = next((dst for old, dst in originals if same_view(old, src)), None)
        dst = alias if alias is not None else shift_image(
            src, spec, seed=stable_seed(seed, *episode_key, int(obs.step), camera, spec.canonical))
        images[camera] = dst
        originals.append((src, dst))

    def mirror(value, path="raw"):
        if isinstance(value, dict):
            return {key: mirror(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [mirror(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(mirror(item, f"{path}[{i}]") for i, item in enumerate(value))
        if not isinstance(value, np.ndarray) or value.ndim < 3 or value.shape[-1] != 3:
            return value
        if value.ndim == 4:
            return np.stack([mirror(frame, f"{path}[{i}]") for i, frame in enumerate(value)])
        # Memory/stride identity distinguishes independently rendered cameras
        # with equal pixel values. Value matching is only a fallback for
        # runners that copied/rotated raw images before creating observations.
        for src, dst in originals:
            if same_view(src, value):
                return dst.copy()
            if same_view(src[::-1, ::-1], value):
                return dst[::-1, ::-1].copy()
        candidates = []
        for src, dst in originals:
            if src.shape != value.shape:
                continue
            if np.array_equal(src, value):
                candidates.append(dst)
            if np.array_equal(src[::-1, ::-1], value):
                candidates.append(dst[::-1, ::-1])
        if candidates:
            if all(np.array_equal(candidates[0], item) for item in candidates[1:]):
                return candidates[0].copy()
            raise ValueError(f"ambiguous raw RGB alias at {path}; explicit camera mapping is required")
        if spec.kind == "clean":
            return value.copy()
        raise ValueError(f"unmapped raw RGB image at {path}; adapter requires an explicit alias mapping")

    return replace(obs, images=images, raw=mirror(obs.raw))
