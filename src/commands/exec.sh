#!/usr/bin/env bash

# Stop execution immediately if any command fails
set -e

echo "Starting first script..."
/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/run_UCF_experiment_train_and_checkpoint_256.sh

echo "First script finished. Starting second script..."
/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/run_UCF_experiment_train_and_checkpoint_256_b.sh

echo "Both scripts completed successfully."