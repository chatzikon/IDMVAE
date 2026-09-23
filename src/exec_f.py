import subprocess

scripts = [
    "commands/run_UCF_experiment_train_and_checkpoint_256_b.sh",
    "commands/run_UCF_experiment_train_and_checkpoint_256.sh",
    "commands/run_UCF_experiment_train_and_checkpoint_256_c.sh"
]

for script in scripts:
    print(f"\n=== Running {script} ===")
    subprocess.run(["bash", script], check=True)

print("\nBoth scripts finished successfully.")