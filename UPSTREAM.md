# Provenance and scope of modifications

Source: https://github.com/jiuyixu25/VLAQuantBench

Starting commit: `4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9` (original `master`). Git history is retained. The original code is MIT licensed and its copyright and permission notice remain in `LICENSE`.

VLAQuantBench authors: Jiuyi Xu, Qing Jin, Meida Chen, Song Wang, Yang Sui, and Yangming Shi. Paper: [VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2609.25376). The original README and citation are archived in `docs/UPSTREAM_README.md` and `docs/UPSTREAM_CITATION.cff`.

## Inherited work

`vlaquantbench/`, original configurations, scripts, tests, diagnostics, figures and `results/` come from upstream unless the diff indicates modifications. Historical data remain upstream data; they are not new VLAQuantShift experiments. The archived README is verbatim; its relative links were written for the upstream repository root.

## New work

`vlaquantshift/`, `configs/shift/`, `tests/test_shift_*.py`, `docs/shift_*.md`, `paper/shift_*`, `verification/`, and the new CI workflow implement or document the extension. Changes to `vlaquantbench/cli.py`, `benchmarks/base.py`, and `results.py` add intervention hooks, pairing provenance, calibration controls and resume compatibility. Original quantization mathematics and model adapters are retained.

Inspect the exact diff with `git diff 4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9..main`. Package metadata and the landing README describe the extension; original authorship remains in this file, Git history, citation and license. Attribution does not establish scientific novelty: new claims require controlled experiments and related-work comparisons.

The extension redistributes no pretrained checkpoints, external benchmark assets or third-party paper text. Model/simulator stacks retain their own licenses and requirements. This project is independent of the upstream authors.
