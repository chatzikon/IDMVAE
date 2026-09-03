#!/usr/bin/env python3

import os
import subprocess
import sys

SCRIPT = "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/run_UCF_experiment_train_and_checkpoint_256.sh"




env = os.environ.copy()

result = subprocess.run(
    ["bash", SCRIPT],
    env=env,
)
