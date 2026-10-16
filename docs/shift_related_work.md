# Related work and overlap audit

Verified against primary paper pages and author repositories on 2026-10-16. Descriptions below are paraphrases. This is a scoped literature check, not proof of exhaustive novelty. The new repository develops upstream VLAQuantBench with attribution; inherited code and results do not become original contributions by being renamed.

## Closest overlap and candidate distinction

| Source | Established overlap | Consequence for VLAQuantShift |
| --- | --- | --- |
| [VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2609.25376) | Closed-loop format/scope comparisons, calibration controls, fixed-observation replay, action fidelity and task-cluster uncertainty. | Reuse and cite the framework. Target explicit four-cell shift contrasts and preserved episode pairing; no claim of first closed-loop quantization benchmark. |
| [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.20309) | Selective quantization with attention temperature and output-energy calibration. | Calibration robustness can be evaluated; renaming these corrections would not establish novelty. |
| [QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization](https://arxiv.org/abs/2602.03782) | Action-sensitive channel allocation, mixed precision and cumulative action sensitivity. | An action-MSE precision selector is a baseline. Cumulative action error alone is not a new contribution. |
| [Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning](https://arxiv.org/abs/2610.05745) (PIVOT-Q) | Quantized VLA robustness under LIBERO-Plus shifts and selective on-policy correction of vulnerable states. | This is the closest overlap with a robustness story. Emphasize audited interaction estimation and evaluation rather than claiming to discover quantized-policy vulnerability. Any empirical distinction still needs evidence. |
| [DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models](https://arxiv.org/abs/2604.11572) | Trajectory drift, representation compensation, motion-driven mixed precision. | Drift awareness and trajectory-based allocation are existing ideas. An offline decomposition is a diagnostic, not an untested new quantizer. |
| [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models](https://arxiv.org/abs/2510.13626) | VLA perturbation evaluation across visual, geometric, language and state changes. | Use official perturbations for a matched benchmark comparison when integrated. Local observation corruptions must be named and documented separately. |
| [Calibrate Where You Deploy](https://github.com/parastoopil/on-policy-calibration-vla) | Public repository investigating on-policy calibration, paired evaluation and state-distribution diagnostics; reports a negative result. | Cite as a software/research artifact, not a verified peer-reviewed paper. Its reported measurements have not been independently reproduced here. On-policy calibration and paired trajectories are not unoccupied ideas. |

The upstream [limitations section](https://arxiv.org/html/2609.25376v1#S7) explicitly distinguishes fixed-observation replay from the changed state distribution visited by a quantized policy and notes that its task bootstrap does not retain episode pairing. These are concrete extension opportunities, but closing a software limitation does not by itself establish a publishable scientific contribution.

## Additional necessary comparisons

- [DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models](https://arxiv.org/abs/2603.07904) studies dynamic activation precision and compensation for accumulated rotation error in its current v3. Runtime switching or phase-dependent precision must be compared with it if added. Earlier versions used a kinematics-guided framing; pin the cited version.
- [ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2605.24011) uses action-guided mixed precision and scale optimization. Budgeted allocation must distinguish itself from this and QVLA.
- [HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.13710) uses policy-sensitive binarization. It belongs in method coverage when a matched implementation is available.
- [Mixed-Precision Quantization for Deep Vision Models with Integer Quadratic Programming](https://arxiv.org/abs/2307.05657) (CLADO, current v2 title) accounts for cross-layer quantization dependencies. Cross-layer interaction is established outside robotics as well. The earlier v1 title was *Mixed-Precision Quantization with Cross-Layer Dependencies*.
- [VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2412.18194), with [official code](https://github.com/OpenMOSS/VLABench), supplies a broader task setting. VLAQuantBench already evaluates X-VLA there; simply connecting the same benchmark is not novel.

## Claim boundaries

The proposed excess-degradation statistic is a standard factorial interaction contrast. The cross-policy/state action identity is elementary algebra. Neither should be advertised as a new estimator or theory. Their value here is a reproducible, explicitly paired protocol with leakage checks and falsifiable questions about precision configurations.

A defensible future empirical claim could concern a reproducible interaction or configuration-rank reversal across independent models, tasks and calibration sets that prior comparisons do not establish. Such a claim requires the actual four-cell measurements and appropriate uncertainty. It must survive comparisons against PIVOT-Q's robustness findings and DA-PTQ's drift framing.

The initial implementation provides observation interventions through inherited runners and offline action-array analysis. It does not yet supply physical lighting changes, an official LIBERO-Plus integration, simulator branching, matched stochastic cross-policy replay, or new real-robot measurements. A toy fixture verifies software integration and cannot establish a VLA effect, speedup, or publication-level novelty.

All sources have corresponding entries in [the bibliography](../paper/shift_references.bib). Pin paper versions and code revisions in the final experiment manifest; current arXiv titles and authors can change between versions.
