"""Exercise public commands across process boundaries, including resumability."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_demo_analyze_and_resume_in_fresh_processes(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    output = tmp_path / "fixture"
    def run(*args):
        subprocess.run([sys.executable, "-m", "vlaquantshift.cli", *map(str, args)],
                       env=env, check=True, capture_output=True, text=True)
    run("demo", "--out", output, "--episodes", 2, "--bootstrap", 40)
    report = json.loads((output / "report.json").read_text())
    assert report["robotics_claim"] is False
    assert len(report["comparisons"]) == 4
    run("analyze-plan", output / "plan.json", "--out", tmp_path / "analysis.json", "--bootstrap", 40)
    analyzed = json.loads((tmp_path / "analysis.json").read_text())
    assert analyzed["comparisons"] == report["comparisons"]
    plan = json.loads((output / "plan.json").read_text())
    first = plan["jobs"][0]
    before = Path(first["out"]).read_bytes()
    run("execute", output / "plan.json", "--job-id", first["id"])
    assert Path(first["out"]).read_bytes() == before
