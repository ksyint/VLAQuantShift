# VLAQuantShift

**Paired quantization-by-distribution-shift experiments for vision-language-action policies.**

A research extension of [VLAQuantBench](https://github.com/jiuyixu25/VLAQuantBench), retaining its Git history, adapters, quantizers, runners and attribution. Starting upstream commit: `4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9`. This project is independent of the upstream authors.

**Status:** executable research software and a manuscript protocol draft. New measurements shipped here are synthetic integration checks. No new VLA/robot accuracy, SOTA, compression speedup, or publication-level novelty is claimed. Historical `results/` belong to the original authors and original paper.

한국어: 기존 코드를 출처와 함께 발전시키는 연구 저장소입니다. 핵심 질문은 **환경 변화가 있을 때 양자화의 추가 성능 손실이 얼마나 커지는가**입니다. 실제 모델 실험을 마친 논문이 아니며, 새 수치는 실행 검증용 합성 환경 결과입니다.

## What changes

For each task/initial state/seed, collect four outcomes:

| | Clean observations | Shifted observations |
| --- | --- | --- |
| Released precision | FP-clean | FP-shift |
| Quantized precision | Q-clean | Q-shift |

```text
excess degradation = (FP-shift − Q-shift) − (FP-clean − Q-clean)
```

Success rates are fractions; multiply by 100 for percentage points. Positive values mean the quantization penalty increases under the specified shift. Inspect all four rates: shared catastrophic failure can yield zero interaction, and floor/ceiling effects can make negative interactions misleading as a robustness claim.

New code provides:

- Deterministic image interventions before model preprocessing, including raw RGB alias handling without mutating simulator images.
- Episode-identity-based seeds, calibration/evaluation index separation, calibration cache separation/content hashes, and incompatible-resume rejection in `vqb run`.
- Validated factorial manifests with explicit coverage, stable hashes, paired comparison definitions and shell-free execution.
- Strict four-arm analysis: missing/duplicate episodes and mismatched recipes are errors; equal-task effects and paired task-cluster bootstrap intervals are reported.
- Offline four-action-array decomposition into direct quantization discrepancy, response to changed state/history, and interaction, including signed cross terms.

These are implementation contributions. Factorial contrasts, action sensitivity, trajectory drift and robust VLA evaluation have substantial prior work; see [related work](docs/shift_related_work.md). Simulator branching and collection of crossed action arrays remain future work.

## Install and verify

Use the appropriate model environment following the inherited [environment scripts](scripts/env). Different VLA/simulator stacks remain incompatible. This extension requires Python 3.10+ and an editable checkout: model configurations are repository resources.

```bash
git clone https://github.com/ksyint/VLAQuantShift.git
cd VLAQuantShift
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
CUDA_VISIBLE_DEVICES='' vqs demo --out artifacts/demo --episodes 8
```

The demo uses a small image/goal-conditioned linear controller and the **actual inherited RTN quantizer, runner, writer and new analysis**. It does not load a VLA or simulate a real robot. [Verification records](verification/README.md) describe executed checks and limits.

This distribution installs `vqs`, `vqb` and both Python packages. Use a separate environment if `vlaquantbench` is already installed: both distributions otherwise own the same import path and `vqb` entry point.

## Plan and run a model study

```bash
vqs plan configs/shift/libero_pilot.yaml \
  --out artifacts/pilot-plan.json --print-commands
vqs execute artifacts/pilot-plan.json --gpus 6,7
vqs analyze-plan artifacts/pilot-plan.json \
  --out artifacts/pilot-analysis.json --bootstrap 5000
```

This specification expands to **48 cells / 9,600 evaluation episodes**, excluding calibration. Planning generates a manifest; execution requires the pi0.5 environment, LIBERO/assets and checkpoint. `--gpus 6,7` maps physical GPU 6 to `cuda:0` and GPU 7 to `cuda:1`; the supplied plans use one worker on `cuda:0`. Select individual cells with repeated `--job-id` arguments.

`checkpoint: lerobot/pi05_libero_finetuned_v044` is a repository identifier, not an immutable weight revision. Before scientific collection, use a pinned local snapshot and record its hash and simulator/assets revisions. The pilot is not a complete preregistration or held-out selection/test split.

Direct use of the modified original CLI:

```bash
CUDA_VISIBLE_DEVICES=6,7 vqb run \
  --model pi05 --benchmark libero --suite libero_spatial \
  --checkpoint /path/to/pinned/pi05_libero --device cuda:0 \
  --preset W4A8 --scope e2e --tasks 0,1 \
  --episodes-from 0 --episodes 20 --seed 0 \
  --eval-shift gaussian_noise:0.05 --calib-shift clean --shift-seed 173 \
  --out shift_runs/pi05-w4a8-noise.jsonl
```

`--episodes` is the **exclusive endpoint**: `--episodes-from 5 --episodes 20` requests 5–19. Every comparison needs corresponding FP/Q clean/shift files. `--eval-shift clean` enables identity-seeded execution and provenance for clean arms. Omitting it retains legacy behavior; legacy files cannot be silently mixed into paired analyses. The extension currently audits RTN runs; inherited alternative quantizers remain available through legacy `vqb`.

The [calibrated config](configs/shift/libero_calibrated.yaml) collects calibration on indices 20–21 and evaluates 0–19. A different calibration source needs a separate config, held fixed within each quartet. Index separation does not prove task-disjointness or detect renamed duplicate assets. Selection/test separation beyond these checks remains an explicit protocol responsibility.

## Observation conditions

| String | Exact semantics |
| --- | --- |
| `clean` | Copy without corruption |
| `brightness:0.2` | Add 0.2 × 255 to uint8 RGB, clip to [0,255] |
| `gaussian_noise:0.05` | Gaussian pixel noise with sigma 0.05 × 255, then clip |
| `occlusion:0.15` | Zero a random rectangle with approximately 15% image area |
| `camera_dropout:0.2` | Independently blank each camera/call with probability 0.2 |

Seeds depend on condition, task, episode, step/call and camera without consuming policy RNG. Shared-memory aliases share the intervention; unmapped or ambiguous raw RGB aliases fail explicitly. A chunked policy may consume images only when replanning. These are sensor corruptions, not physical scene/lighting changes or an official LIBERO-Plus implementation. Severity is restricted to [0,1].

Episode seed alignment improves reproducibility across resume/calibration order, but does not establish identical noise at each inference call when histories or RNG consumption differ. Traces contain compact observation fingerprints and action variation, not replayable trajectories.

The inherited VLABench runner retains native tracks and per-task episode caps. The new analysis is binary success; native progress-score analysis, semantic-shift contrasts and real VLABench execution are not validated here. Nonzero shifted VLABench episode starts are rejected. CALVIN requires one episode per chain/task; activation calibration there is rejected until task-chain split support exists.

## Analyze data

```bash
vqs analyze --fp-clean fp-clean.jsonl --q-clean q-clean.jsonl \
  --fp-shift fp-shift.jsonl --q-shift q-shift.jsonl \
  --bootstrap 5000 --out comparison.json
vqs cross-replay crossed-actions.npz --out action-decomposition.json
```

NPZ keys: `fp_on_fp`, `q_on_fp`, `fp_on_q`, `q_on_q` with shape `[T,D]`, optionally positive per-dimension `scale`. Inputs must already share units, valid timesteps, histories and intended noise coupling. The analyzer verifies algebra/numerics, not how actions were collected.

Task bootstrap intervals are conditional on observed episodes/calibration. Multi-seed plans return separate contrasts, not a pooled or crossed uncertainty estimate. Secondary intervals are descriptive without familywise correction. Fake-quantized accuracy runs do not establish packed storage, lower VRAM or integer-kernel acceleration.

## Research and attribution

- [Research plan](docs/shift_research_plan.md) and [related work](docs/shift_related_work.md)
- [Manuscript protocol draft](paper/shift_protocol.tex) and [bibliography](paper/shift_references.bib)
- [Upstream/change ownership](UPSTREAM.md)
- [Original README, archived verbatim](docs/UPSTREAM_README.md) and [original citation](docs/UPSTREAM_CITATION.cff)

Cite [VLAQuantBench](https://arxiv.org/abs/2609.25376) for its code, evaluation framework and historical results, together with this software version for the extension. Other motivating work is cited in the research documents. The original MIT notice remains in [LICENSE](LICENSE), with a separate copyright line for modifications.
