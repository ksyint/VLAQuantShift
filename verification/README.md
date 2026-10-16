# Verification evidence

This directory is reserved for reproducible software verification, not VLA or robot performance claims. The synthetic controller uses fixed linear weights and RGB-encoded state; it calls the original VLAQuantBench RTN quantizer and closed-loop runner.

Run the checks from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
CUDA_VISIBLE_DEVICES=6 python -m pytest -q
CUDA_VISIBLE_DEVICES='' vqs demo --out artifacts/verification_cpu --episodes 8 --bootstrap 500
CUDA_VISIBLE_DEVICES=6 vqs demo --out artifacts/verification_gpu6 --device cuda:0 --episodes 4 --bootstrap 500
python scripts/verify_cells.py
```

The last command audits historical upstream records and does not create new experiments. New demo runs record their source revision, runtime and synthetic status. Timing in this fixture includes Python instrumentation overhead and is not a model/hardware performance comparison.

## Executed checks (2026-10-16)

- CPU tests: **175 passed, 5 GPU-only tests skipped**.
- CUDA 6 tests: **180 passed**. Both suites emit 224 inherited multiprocessing/fork deprecation warnings; full summaries are saved here.
- CPU synthetic integration: **288 episodes across 9 cells**.
- Physical GPU 6 / NVIDIA B200 synthetic integration: **144 episodes across 9 cells**. CUDA 7 was not needed.
- Historical upstream audit: **417 cells verified**. These are original VLAQuantBench data, not our experiments.
- Editable package and both `vqs`/`vqb` entry points installed successfully. The 48-cell pilot and 24-cell calibrated manifests validate; real-model evaluation was not launched.

Measurement source: commit `95dd37a` (full SHA in [summary.json](summary.json)). Every archived run records a clean source tree at collection. JSONL bytes are unchanged from collection. Archive manifests use repository-relative paths, and reports were recomputed against these relocated records.

```bash
vqs analyze-plan verification/synthetic_cpu/plan.json --out artifacts/recomputed-cpu.json --bootstrap 500
vqs analyze-plan verification/synthetic_gpu6/plan.json --out artifacts/recomputed-gpu6.json --bootstrap 500
```

[CPU report](synthetic_cpu/report.json), [GPU report](synthetic_gpu6/report.json), and [record SHA256s](summary.json) are included. Demo differences, including success floors, are characteristics of the synthetic fixture and cannot support a claim about pretrained VLAs.

Real LIBERO/VLABench checkpoints, simulator assets and their full experiments remain to be collected. The LaTeX draft was checked for citation-key consistency but was not compiled because TeX tooling is absent. GitHub Actions performs the CPU checks separately; local logs here do not assert a remote CI outcome.
