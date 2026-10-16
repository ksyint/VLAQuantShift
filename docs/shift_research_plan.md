# VLAQuantShift research plan

Status: protocol and implementation candidate, 2026-10-16. This document specifies future scientific experiments; it does not report new VLA or robot performance. Toy fixtures establish software behavior only. Publication novelty and positive experimental results are not assumed.

한국어 요약: 원 VLAQuantBench의 코드를 출처와 함께 유지하고, **양자화가 환경 변화에 대한 성능 하락을 추가로 키우는지**를 동일한 초기 조건의 네 실험으로 측정한다. 현재 목표는 재현 가능한 실험 도구와 검증 가능한 연구 가설이다. 실제 VLA 실험, 통계 검증, 기존 방법 비교를 마친 뒤에만 논문의 결과와 기여를 확정한다.

## Attribution and intended contribution

This repository develops [VLAQuantBench](https://github.com/jiuyixu25/VLAQuantBench), starting from upstream commit `4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9`. Its upstream code, MIT license, citation, model adapters, quantizers, evaluators, and historical results retain their attribution. Historical upstream results are evidence for the upstream paper, not new VLAQuantShift measurements. New source and records must remain identifiable through commits and explicit provenance.

The candidate contribution is an auditable protocol for quantization-by-shift interactions: paired outcome contrasts, calibration/evaluation separation, deterministic observation interventions, and an offline policy-by-state action decomposition. This is not a claim to have invented closed-loop PTQ evaluation, robust VLA evaluation, mixed precision, difference-in-differences, or action sensitivity. See the [overlap matrix](shift_related_work.md).

The main question is: **Does a precision configuration that preserves clean success also preserve robustness, and does the answer change with calibration source or action-execution horizon?** A secondary question asks whether configuration rankings based on clean action error transfer to held-out shifts. Both can yield null or negative findings.

## Primary estimand

For one checkpoint and one quantization recipe, run four conditions on the same task and initial-state identities:

| Condition | Policy | Observation condition |
| --- | --- | --- |
| FP-clean | Released precision | Identity transform |
| Q-clean | Quantized | Identity transform |
| FP-shift | Released precision | Prespecified shift |
| Q-shift | Quantized | The same shift |

For task `j`, matched episode `i`, and binary success `Y`, define

```text
e[j,i] = (Y_fp_shift - Y_q_shift) - (Y_fp_clean - Y_q_clean)
E = mean_over_tasks(mean_over_matched_episodes(e[j,i]))
```

Positive `E` means that the quantization penalty is larger under the shift. Report `100 * E` in percentage points, every cell's success rate, and both clean and shifted quantization penalties. Negative values are possible. A large shift failure shared by FP and Q can coexist with zero excess degradation; therefore `E` must never replace the four raw outcomes. If both shifted policies fail every episode, `E` equals the negative clean quantization penalty: an apparently favorable interaction can reflect a shared success floor rather than robustness. Clean success ceilings similarly conceal action-quality differences. Report the paired FP/Q success/failure patterns and task-level rates; do not exclude floor/ceiling tasks after observing outcomes. This factorial contrast is descriptive under the specified intervention and checkpoint, not a general causal claim about real-world disturbances.

Preregister one primary model/checkpoint, suite, quantizer, bit setting, shift type and severity before seeing test results. Proposed primary contrast: pi0.5 (`lerobot/pi05_libero_finetuned_v044`), LIBERO-Spatial, end-to-end RTN W4, `brightness:0.2`, evaluation seed 0, compared with released precision. This matches a cell in the executable pilot configuration. Brightness is **additive**: add `0.2 * 255` to every uint8 RGB channel, clip to `[0, 255]`, then cast to uint8; it is not a multiplicative factor. Pin the resolved checkpoint revision, selected modules, transform semantics, and suite revision before launch. Use the equal-task mean of `e` as the primary metric, with a two-sided 95% task-cluster bootstrap interval. Seeds 1 and 2 and OpenVLA-OFT replication are secondary until a joint seed analysis is prespecified and implemented. Other cells are secondary; do not choose the headline after looking at them. Change this proposal only before test execution and record the revision. An explicit Hugging Face checkpoint name in the current config is not an immutable model revision.

## Pairing and uncertainty

Pair using an explicit tuple containing benchmark/suite, task ID, initial-state ID, evaluation seed, shift definition and severity, and replicate identity. Preserve all four cell values within each resampled tuple. Reject duplicate cells and fail on incomplete quartets by default; report any explicitly permitted exclusions and their causes. Never treat a missing, crashed, or unfinished episode as failure or success without a preregistered policy.

Resample tasks with replacement, carrying their complete paired episode records, and recompute the equal-task statistic for at least 5,000 bootstrap draws using a recorded seed (`--bootstrap 5000`; the current CLI default is 2,000). The implemented analyzer resamples **tasks only**, conditional on their recorded episodes. A second stage resampling matched episode identities is a proposed extension, not current behavior. The current `analyze-plan` produces a separate contrast for each seed, without a joint multi-seed confidence interval. These intervals do not include new within-task episode sampling, calibration variation, or model-training uncertainty. Few task clusters require cautious interpretation and reporting of all task-level contrasts. A collapsed interval with one task or identical observed task effects is not evidence of population certainty.

For calibrated methods, repeat calibration with at least three separately collected calibration sets. Report each calibration replicate and its task-paired interval separately, plus the spread across replicates; do not pool them as independent episodes. Current configurations reuse calibration initial states 20–21 on the first suite task, and the same `seed` controls both calibration and evaluation. Therefore the three configured seeds do not implement an independently crossed calibration-set-by-evaluation-seed experiment. Separate configuration files with distinct calibration episode ranges are needed to vary calibration sets; an independent calibration seed and crossed task-by-calibration analysis remain extensions.

Matched initial states and seeds do not automatically establish matched stochastic noise. Record whether the adapter uses explicit per-step common random numbers, deterministic inference, or only an initial RNG seed. Until an adapter audit establishes stronger coupling, describe runs as matched by episode identity and seed. Differences in random-number consumption, caches, reset behavior, and early stopping can weaken coupling.

The implemented corruption RNG is derived separately from `shift_seed`, task/episode identity, control step, camera and shift definition. The sample plans hold `shift_seed` fixed across evaluation seeds. Thus those seeds repeat policy randomness under a fixed corruption schedule, not independent corruption realizations. Severity-specific seeds also do not guarantee nested masks/noise across severity levels. Claims about a distribution of corruptions need separately prespecified shift-seed replication.

## Calibration, selection, and evaluation splits

The final scientific study requires three immutable manifests:

1. **Calibration:** observations used to fit clipping, smoothing, or quantizer parameters. Record source policy, task IDs, episode IDs, shift conditions, collection seed, and data digest.
2. **Selection:** tasks and episodes used to choose bit budgets, protected modules, shift thresholds, or a precision configuration.
3. **Evaluation:** untouched task/episode identities and prespecified shifts used for the reported estimates.

Disallow episode overlap between all roles. For claims about task transfer, also require task-disjoint splits; holding out only initial-state indices supports episode generalization, not unseen-task generalization. Suite labels alone do not establish disjointness. Preserve asset/task identity across renamed tasks and transformed views of the same episode. **Current enforcement is narrower:** the plan/runner rejects overlapping calibration/evaluation episode-index ranges; it does not implement three-role manifests, a selection split, task-disjointness, or renamed-asset identity checks. These are protocol requirements to implement or audit externally before a final study, not completed capabilities.

Freeze quantizer parameters after calibration. Use identical frozen parameters in Q-clean and Q-shift when estimating zero-shot robustness. The cache key includes calibration condition and execution settings, while evaluation condition is intentionally excluded. The extension records and compares the SHA256 of calibration cache contents. Matching cache metadata/digests does not verify immutable checkpoint weights; final experiments must also retain and verify checkpoint artifact digests. Recalibration on shifted observations is a distinct adaptation arm with a separately recorded calibration budget. Current collection uses an unquantized policy and one calibration observation transform per plan; mixing calibration sources or collecting on-policy quantized states requires additional implementation. Never update scales or choose precision on evaluation observations unless the protocol explicitly labels a test-time adaptation method and constrains its resources.

## Action diagnostics: offline first

Let `s_f` and `s_q` denote paired FP and quantized histories/states at the same control index. Evaluate four action arrays using matched inference randomness and explicitly controlled policy memory:

```text
a_ff = pi_fp(s_f)       a_qf = pi_q(s_f)
a_fq = pi_fp(s_q)       a_qq = pi_q(s_q)
D = a_qf - a_ff
S = a_fq - a_ff
I = a_qq - a_fq - a_qf + a_ff
T = a_qq - a_ff = D + S + I
```

`D` is quantization discrepancy on the reference history, `S` is the FP response to the different history, and `I` is the policy-by-history interaction. This is an exact algebraic decomposition. The first release analyzes supplied arrays; **it does not implement simulator snapshot/restore or collect these crossed evaluations from real VLA rollouts**. Correct cross evaluation will require an adapter-specific treatment of recurrent state, action queues, visual history, noise, action normalization, and chunk execution.

Validate vector closure and report norms with signed inner-product terms, since `||T||` is not `||D|| + ||S|| + ||I||`. Compare only common valid timesteps and record the number omitted after either trajectory terminates. The current analyzer accepts four finite `T × D` arrays of equal shape and optional dimension scales; the caller must perform alignment, censoring, episode grouping and calibration-only scale fitting. It does not collect or verify this provenance. Do not zero-pad terminated trajectories, treat timesteps as independent episodes, or infer physical damage from an action norm. Define translation, rotation and gripper units before aggregating dimensions. Any normalization must be fitted on calibration data and then frozen.

## Planned experiment matrix

| Axis | Initial scientific study | Expansion after validation |
| --- | --- | --- |
| Models | OpenVLA-OFT and pi0.5, pinned released checkpoints | X-VLA or another architecture |
| Suites | LIBERO-Spatial and LIBERO-Long, checkpoint/task identities explicit | VLABench task-progress evaluation; keep its metric separate |
| Precision | FP, RTN W4, W4A8; one protected-output configuration | Matched available VLA PTQ methods after reproduction |
| Shifts | Implemented identity, additive brightness, seeded Gaussian noise, rectangle occlusion and camera dropout; one primary severity | Contrast, prespecified severity sweeps and official LIBERO-Plus scene perturbations |
| Calibration | None for RTN; separate plans for disjoint calibration replicates using one clean or shifted FP collection source | Independent calibration/evaluation seeds and mixed-source calibration at equal sample budgets |
| Execution | Released action-chunk convention held fixed | Prespecified execution-chunk ablation |
| Replication | At least 20 initial states per held-out task and 3 evaluation seeds if affordable | Increase using an external pilot's precision/power calculation |

The numerical budgets above are planning targets, not achieved coverage or a power guarantee. Shared initial conditions across seeds are repeated measures. Hold out entire tasks for selection; preregister eligible tasks and whether within-suite test means cover only that held-out subset. The shipped pilot lists all ten Spatial tasks for convenient coverage testing; before collecting a scientific pilot, restrict it to tasks separate from the final test set. A small pilot uses separate tasks and never contributes to final estimates.

Apply observation transforms at a documented boundary before policy preprocessing to all intended camera streams. Record value range, dtype, clipping, stochastic seed derivation, and transform order. An image brightness transform is an observation corruption; it is not physically equivalent to changing simulator lighting. The first release's transforms are not an official LIBERO-Plus reproduction. Actual scene perturbations require a separately pinned integration.

Use inherited runners only when their reset, episode identity, action normalization, and task metrics are verified. Keep the FP and Q branches' environment settings identical. Fake quantization establishes accuracy effects; theoretical bit counts are not measured VRAM, packed storage, or acceleration. Report actual device/runtime measurements separately if implemented. On this workstation, any GPU execution is restricted to CUDA devices 6 and 7.

## Required ablations and decision rules

- **Zero-shift control:** identity versus identity should give a zero interaction under deterministic, identical execution; any residual motivates a coupling audit.
- **No-quantization control:** independently executed FP arms test reset/RNG reproducibility rather than quantization.
- **Pairing control:** compare paired and unpaired uncertainty only as an analysis audit; use the preregistered paired estimate for claims.
- **Calibration budget:** vary source while keeping examples, collection cost and quantizer recipe fixed. Distinguish source shift from quantity.
- **Scope control:** compare uniform RTN with protected output modules. A selector based on action MSE is a reference baseline, not a new QVLA-like method claim.
- **Horizon control:** change executed action count separately from predicted chunk length, keeping the checkpoint fixed.
- **Shift control:** vary severity and camera coverage without selecting the best-looking settings after evaluation.

Report null results and reversals. Do not interpret an interval containing zero as equivalence; a future equivalence claim needs a prespecified tolerance and corresponding test. Multiple secondary contrasts require either multiplicity-aware inference or explicit descriptive labeling.

## Staged completion criteria

1. **Instrument:** pass toy fixtures for pairing, incomplete-cell rejection, split leakage, identity transforms, action-decomposition closure and reproducible summaries. These are software checks.
2. **Adapter pilot:** run released FP and a small RTN condition on separate pilot tasks; verify transforms, module coverage, initial states, seed behavior and logs. Record download/runtime blockers.
3. **Freeze:** commit the full preregistration, hashes and manifests before final test collection. Select the target precision budget using selection data only.
4. **Collect:** execute all four cells for every retained identity and independent calibration replicate. Save raw records and failures; resume without duplicate episodes.
5. **Analyze:** generate the primary estimate, task-level distribution, all four raw success rates and audited secondary contrasts from immutable logs.
6. **Write:** replace pending manuscript cells only with traceable measurements. Recheck related work, especially PIVOT-Q and DA-PTQ, and state any unresolved novelty overlap.

The accompanying [manuscript](../paper/shift_protocol.tex) is a protocol draft. It becomes an empirical paper only after these stages establish a substantive finding beyond the existing literature.
