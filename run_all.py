"""Run the CohortLearn worked example and two validation checks.

main       : worked example, results saved to outputs/
validation : hazard outcome model, true HR = exp(0.5) = 1.65 (console only)
null       : hazard outcome model, true HR = 1.00 (console only)

The two checks write to checks/, which is not kept, so outputs/ holds only the
worked example.

Reproducibility
The maths libraries pick CPU-specific code paths at runtime, which changes the
last digits of floating-point results between machines. On Intel and AMD
processors, this script forces one generic code path and a single thread, so
Docker, conda and plain virtual environments all give the same numbers. The
same settings are used inside the Docker image.
"""

import os
import platform
import subprocess
import sys

RUNS = [
    ("main",       ["--outcome-model", "logistic", "--true-effect", "0.5"], "outputs"),
    ("validation", ["--outcome-model", "hazard",   "--true-effect", "0.5"], "checks/validation"),
    ("null",       ["--outcome-model", "hazard",   "--true-effect", "0.0"], "checks/null"),
]

PINNED_X86 = {
    "OPENBLAS_CORETYPE": "Nehalem",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NPY_DISABLE_CPU_FEATURES": "X86_V3 X86_V4 AVX512_ICL AVX512_SPR",
}


def run_env():
    """Environment for the child processes, with the pinned maths settings."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    if platform.machine().lower() in ("x86_64", "amd64"):
        env.update(PINNED_X86)
    else:
        print(f"Note: {platform.machine()} processor detected. The pinned maths "
              "settings apply to Intel and AMD processors only, so the last "
              "digits may differ from the reference results. Use the Docker "
              "image for identical numbers.", flush=True)
    return env


env = run_env()
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
        subprocess.run(cmd, check=True, env=env)
    if out_dir != "outputs":
        with open(os.path.join(out_dir, "results_summary.txt"), encoding="utf-8") as f:
            text = f.read()
        main = text[text.index("6. Main result"):text.index("7. Unmeasured")]
        print(f"\n--- {name} check ---\n{main.strip()}", flush=True)

print("\nAll runs completed. Worked example saved to outputs/")
