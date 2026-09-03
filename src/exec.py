#!/usr/bin/env python3

import os
import subprocess
import sys

SCRIPT = "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/functions_post_eval/run_UCF_denoiser_train.sh"

DATA_PATHS = [
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA Image Dataset/processed/pregen_4x32x32_1x/IDMVAE_07-29_0_ep15_release",
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA Image Dataset/processed/pregen_4x32x32_1x/IDMVAE_07-29_0_ep45_release",
]

for i, data_path in enumerate(DATA_PATHS, start=1):
    print("=" * 60)
    print(f"Run {i}")
    print(f"DATA_PATH = {data_path}")
    print("=" * 60)

    env = os.environ.copy()
    env["DATA_PATH"] = data_path

    result = subprocess.run(
        ["bash", SCRIPT],
        env=env,
    )

    if result.returncode != 0:
        print(f"\nRun {i} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\nAll runs completed successfully.")