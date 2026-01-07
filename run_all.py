import subprocess
import sys

scripts = [
    "model_mlp.py",
    "model_rbf_scratch.py",
    "model_som.py",
    "model_rnn_elman.py",
    "model_rnn_jordan.py",
    "model_fuzzy_sugeno.py",
    "model_probabilistic_bayesian_ridge.py",
    "ga_optimize_mlp.py",
    "compare_all_models.py",
]

for s in scripts:
    print(f"\n=== Running {s} ===")
    subprocess.run(
        [sys.executable, s],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

print("\n=== ALL SCRIPTS FINISHED SUCCESSFULLY ===")
