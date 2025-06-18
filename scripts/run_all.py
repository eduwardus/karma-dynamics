# run_all.py
import subprocess

scripts = [
    "scripts/seirs_karma.py",
    "scripts/enlightenment.py",
    "scripts/three_roots.py",
    "scripts/five_poisons.py",
    "scripts/realms_attractors.py",
    "scripts/network_karma.py",
    "scripts/stochastic_extensions.py"
]

for script in scripts:
    print(f"\nEjecutando {script}...")
    subprocess.run(["python", script])
