import subprocess
import sys

steps = [
    [sys.executable, "scripts/generate_synthetic_data.py", "--n", "60000", "--out-dir", "data"],
    [sys.executable, "scripts/run_pipeline_example.py", "--data-dir", "data", "--out-dir", "outputs"],
]

for cmd in steps:
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)

print("\nAll steps completed. Results in outputs/")