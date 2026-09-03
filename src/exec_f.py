#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE"
)

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "src/commands/functions_post_eval/eval_UCF_denoiser.sh"
)

HIGH_RES_DATA_PATH = (
    PROJECT_ROOT
    / "archive/UCA Image Dataset/processed"
)


# ---------------------------------------------------------------------
# DiT checkpoints that will be evaluated for BOTH sets
# ---------------------------------------------------------------------

CHECKPOINT_STEPS = [
    "0075000",
    "0150000",
    "0225000",
    "0300000",
]


# ---------------------------------------------------------------------
# Two evaluation sets
#
# Change these paths to your exact paths.
# ---------------------------------------------------------------------

EVAL_SETS = [
    {
        "name": "ep15",

        "data_path": (
            HIGH_RES_DATA_PATH
            / "pregen_4x32x32_1x"
            / "IDMVAE_07-29_0_ep15_release"
        ),

        "denoiser_checkpoint_dir": (
            HIGH_RES_DATA_PATH
            / "pregen_4x32x32_1x"
            / "denoiser"
            / "IDMVAE_07-29_0_ep15_release_000-DiT-XL-2"
            / "checkpoints"
        ),

        "resnet_model_args": (
            PROJECT_ROOT
            / "outputs/UCA_baseline/checkpoints"
            / "07-29_0_gpu0_ltCL_TDeval_lw0.1_K1_B64_"
              "Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
            / "args.json"
        ),

        "resnet_checkpoint": (
            PROJECT_ROOT
            / "outputs/UCA_baseline/checkpoints"
            / "07-29_0_gpu0_ltCL_TDeval_lw0.1_K1_B64_"
              "Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
            / "model_15.rar"
        ),
    },

    {
        "name": "ep45",

        "data_path": (
            HIGH_RES_DATA_PATH
            / "pregen_4x32x32_1x"
            / "IDMVAE_07-29_0_ep45_release"
        ),

        "denoiser_checkpoint_dir": (
            HIGH_RES_DATA_PATH
            / "pregen_4x32x32_1x"
            / "denoiser"
            / "IDMVAE_07-29_0_ep45_release_001-DiT-XL-2"
            / "checkpoints"
        ),

        "resnet_model_args": (
            PROJECT_ROOT
            / "outputs/UCA_baseline/checkpoints"
            / "07-29_0_gpu0_ltCL_TDeval_lw0.1_K1_B64_"
              "Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
            / "args.json"
        ),

        "resnet_checkpoint": (
            PROJECT_ROOT
            / "outputs/UCA_baseline/checkpoints"
            / "07-29_0_gpu0_ltCL_TDeval_lw0.1_K1_B64_"
              "Normal_Laplace_b1.0_10.0_40.0_256_256_s2"
            / "model_45.rar"
        ),
    },
]


def run(command, description):
    print("\n" + "=" * 100)
    print(description)
    print("=" * 100)
    print(" ".join(str(x) for x in command))
    print("=" * 100, flush=True)

    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def validate_paths():
    required = [
        EVAL_SCRIPT,
        HIGH_RES_DATA_PATH,
    ]

    for evaluation_set in EVAL_SETS:

        required += [
            evaluation_set["data_path"],
            evaluation_set["denoiser_checkpoint_dir"],
            evaluation_set["resnet_model_args"],
            evaluation_set["resnet_checkpoint"],
        ]

        for step in CHECKPOINT_STEPS:
            required.append(
                evaluation_set["denoiser_checkpoint_dir"]
                / f"{step}.pt"
            )

    missing = [
        path
        for path in required
        if not Path(path).exists()
    ]

    if missing:
        print("\nERROR: Missing paths:\n")

        for path in missing:
            print(f"  {path}")

        sys.exit(1)


def evaluate(
    evaluation_set,
    step,
    use_diffusion_prior,
):

    ckpt = (
        evaluation_set["denoiser_checkpoint_dir"]
        / f"{step}.pt"
    )

    prior_name = (
        "diffusion_prior"
        if use_diffusion_prior
        else "standard_prior"
    )

    # Parent of "checkpoints/"
    denoiser_run_dir = (
        evaluation_set["denoiser_checkpoint_dir"].parent
    )

    output_path = (
        denoiser_run_dir
        / "evaluations"
        / f"step_{step}_{prior_name}"
    )

    command = [
        "bash",
        EVAL_SCRIPT,

        "--high-res-data-path",
        HIGH_RES_DATA_PATH,

        "--data-path",
        evaluation_set["data_path"],

        "--output-path",
        output_path,

        "--ckpt",
        ckpt,

        "--vae",
        "mse",

        "--batch-size",
        "32",

        "--num-workers",
        "0",

        "--num-sampling-steps",
        "250",

        "--seed",
        "0",

        "--resnet_model_args",
        evaluation_set["resnet_model_args"],

        "--resnet_checkpoint",
        evaluation_set["resnet_checkpoint"],

        "--text2img_qzpw",
        "--img2text_qzpw",
        "--img2img_qzpw",
        "--img2img_qwpz",
        "--img_random",
        "--text_random",
    ]

    if use_diffusion_prior:
        command.append(
            "--use_diffusion_prior"
        )

    run(
        command,
        description=(
            f"Dataset/model : {evaluation_set['name']}\n"
            f"DiT step      : {step}\n"
            f"Prior         : {prior_name}\n"
            f"Output        : {output_path}"
        ),
    )


def main():

    validate_paths()

    total = (
        len(EVAL_SETS)
        * len(CHECKPOINT_STEPS)
        * 2
    )

    print(
        "\nPlanned evaluations:"
        f"\n  Sets            : {len(EVAL_SETS)}"
        f"\n  DiT checkpoints : {len(CHECKPOINT_STEPS)}"
        f"\n  Prior variants  : 2"
        f"\n  Total           : {total}\n"
    )

    completed = 0

    for evaluation_set in EVAL_SETS:

        print(
            "\n" + "#" * 100
        )
        print(
            f"STARTING SET: {evaluation_set['name']}"
        )
        print(
            f"DATA_PATH: {evaluation_set['data_path']}"
        )
        print(
            f"RESNET CHECKPOINT: "
            f"{evaluation_set['resnet_checkpoint']}"
        )
        print(
            "#" * 100
        )

        for step in CHECKPOINT_STEPS:

            # Standard Gaussian prior
            evaluate(
                evaluation_set,
                step,
                use_diffusion_prior=False,
            )

            completed += 1
            print(
                f"\nCompleted {completed}/{total}"
            )

            # Learned diffusion prior
            evaluate(
                evaluation_set,
                step,
                use_diffusion_prior=True,
            )

            completed += 1
            print(
                f"\nCompleted {completed}/{total}"
            )

    print(
        "\n"
        + "=" * 100
        + "\nALL 16 EVALUATIONS COMPLETED SUCCESSFULLY"
        + "\n"
        + "=" * 100
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        sys.exit(130)

    except subprocess.CalledProcessError as e:
        print(
            f"\nEvaluation failed with exit code "
            f"{e.returncode}.",
            file=sys.stderr,
        )
        sys.exit(e.returncode)