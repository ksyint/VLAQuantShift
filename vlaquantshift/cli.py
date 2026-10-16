"""VLAQuantShift planning, paired analysis and synthetic integration demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def _write(value, output):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="validate a factorial YAML and write an executable manifest")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--output-root", default="shift_runs")
    p.add_argument("--print-commands", action="store_true")
    p = sub.add_parser("execute", help="execute a validated plan in the current model environment")
    p.add_argument("plan")
    p.add_argument("--gpus", default="6,7")
    p.add_argument("--job-id", action="append")
    p = sub.add_parser("analyze", help="strict paired FP/Q by clean/shift factorial contrast")
    for arm in ("fp-clean", "q-clean", "fp-shift", "q-shift"):
        p.add_argument(f"--{arm}", required=True)
    p.add_argument("--out")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("analyze-plan", help="analyze every complete planned factorial contrast; incomplete cells are errors")
    p.add_argument("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p = sub.add_parser("cross-replay", help="decompose four matched action arrays stored in NPZ (no pickle)")
    p.add_argument("input", help="NPZ keys: fp_on_fp, q_on_fp, fp_on_q, q_on_q; optional scale")
    p.add_argument("--out")
    p = sub.add_parser("demo", help="synthetic integration check, NOT a VLA performance experiment")
    p.add_argument("--out", default="artifacts/synthetic_demo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command == "plan":
        import yaml
        from .protocol import build_plan
        with open(args.config) as fh:
            plan = build_plan(yaml.safe_load(fh), args.output_root)
        _write(plan, args.out)
        print(f"{len(plan['jobs'])} cells; {plan['total_episodes']} evaluation episodes; {len(plan['comparisons'])} contrasts")
        if args.print_commands:
            for job in plan["jobs"]:
                print(shlex.join(job["argv"]))
    elif args.command == "execute":
        from .protocol import execute_plan
        execute_plan(json.loads(Path(args.plan).read_text()), gpu_ids=args.gpus, job_ids=args.job_id)
    elif args.command == "analyze":
        from .analysis import analyze_factorial
        _write(analyze_factorial(args.fp_clean, args.q_clean, args.fp_shift, args.q_shift,
                                 n_bootstrap=args.bootstrap, seed=args.seed), args.out)
    elif args.command == "analyze-plan":
        from .analysis import analyze_factorial
        from .protocol import verify_plan
        plan = verify_plan(json.loads(Path(args.plan).read_text()))
        reports = [{"comparison": comparison, "analysis": analyze_factorial(
            *[comparison[arm] for arm in ("fp_clean", "q_clean", "fp_shift", "q_shift")], n_bootstrap=args.bootstrap)}
            for comparison in plan["comparisons"]]
        _write({"plan_sha256": plan["plan_sha256"], "comparisons": reports,
                "multiplicity": "Per-contrast descriptive intervals; not familywise-corrected. Predeclare primary contrast."}, args.out)
    elif args.command == "cross-replay":
        import numpy as np
        from .analysis import analyze_cross_replay
        with np.load(args.input, allow_pickle=False) as data:
            report = analyze_cross_replay(*[data[k] for k in ("fp_on_fp", "q_on_fp", "fp_on_q", "q_on_q")],
                                          scale=data["scale"] if "scale" in data else None)
        _write(report, args.out)
    elif args.command == "demo":
        from .fixture import run_demo
        report = run_demo(args.out, device=args.device, episodes=args.episodes, n_bootstrap=args.bootstrap)
        print(f"Synthetic integration only: {len(report['comparisons'])} contrasts saved to {args.out}/report.json")


if __name__ == "__main__":
    main()
