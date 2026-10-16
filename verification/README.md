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
