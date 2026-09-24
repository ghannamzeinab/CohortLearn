"""Run the CohortLearn worked example and two validation checks.

main       : worked example, results saved to outputs/
validation : hazard outcome model, true HR = exp(0.5) = 1.65 (console only)
null       : hazard outcome model, true HR = 1.00 (console only)

The two checks write to checks/, which is not kept, so outputs/ holds only the
worked example.
"""

import os
import subprocess
import sys

RUNS = [
    ("main",       ["--outcome-model", "logistic", "--true-effect", "0.5"], "outputs"),
    ("validation", ["--outcome-model", "hazard",   "--true-effect", "0.5"], "checks/validation"),
    ("null",       ["--outcome-model", "hazard",   "--true-effect", "0.0"], "checks/null"),
]

for name, gen_args, out_dir in RUNS:
    data_dir = f"data/{name}"
    steps = [
        [sys.executable, "scripts/generate_synthetic_data.py",
         "--n", "60000", "--out-dir", data_dir, *gen_args],
        [sys.executable, "scripts/run_pipeline_example.py",
         "--data-dir", data_dir, "--out-dir", out_dir],
    ]
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}", flush=True)
    for cmd in steps:
        print("\n>>>", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)
    if out_dir != "outputs":
        with open(os.path.join(out_dir, "results_summary.txt"), encoding="utf-8") as f:
            text = f.read()
        main = text[text.index("6. Main result"):text.index("7. Unmeasured")]
        print(f"\n--- {name} check ---\n{main.strip()}", flush=True)

print("\nAll runs completed. Worked example saved to outputs/")
