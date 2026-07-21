import subprocess

print("Running script1.sh...")
subprocess.run(["bash", "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/"
                        "run_CUB_experiment_train_and_checkpoint_256.sh"])

print("script1.sh finished successfully.")

print("Running script2.sh...")
subprocess.run(["bash", "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/"
                        "run_CUB_experiment_train_and_checkpoint_256_diff.sh"])

print("script2.sh finished successfully.")