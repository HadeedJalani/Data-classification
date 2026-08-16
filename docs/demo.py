from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
print("Project 2 — Data Classification Using AI")
print("Running tests...")
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
print("\nRunning exact assignment demonstration...")
subprocess.run([sys.executable, "-m", "src.main", "--no-tune-k", "--k", "5"], cwd=ROOT, check=True)
print("\nDemo completed successfully.")
