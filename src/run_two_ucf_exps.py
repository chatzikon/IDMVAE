#!/usr/bin/env python3
"""
Run run_UCF_experiment_train_and_checkpoint_256.sh twice:

  Experiment 1:
      cross_mi_loss_scale = 40
      z_alignment_loss_scale = 1000

  Experiment 2:
      cross_mi_loss_scale = 80
      z_alignment_loss_scale = 0

The original shell script is never modified. A temporary copy is created
for each run.

This version supports both:
    --cross_mi_loss_scale 40
and:
    --cross_mi_loss_scale=40

It also supports common shell variable assignments as a fallback.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EXPERIMENTS = (
    {
        "name": "zalign0_mi80",
        "cross_mi": 80.0,
        "z_alignment": 0.0,
    },

    {
        "name": "zalign1000_mi40",
        "cross_mi": 40.0,
        "z_alignment": 1000.0,
    },

)


VALUE_PATTERN = r'(?:"[^"]*"|\'[^\']*\'|\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|[^\s\\]+)'


def replace_cli_option(text: str, option: str, value: float) -> tuple[str, int]:
    pattern = re.compile(
        rf"(?P<option>{re.escape(option)})"
        rf"(?P<sep>\s*=\s*|\s+)"
        rf"(?P<value>{VALUE_PATTERN})"
    )
    return pattern.subn(
        lambda m: f"{m.group('option')}{m.group('sep')}{value}",
        text,
    )


def replace_shell_assignment(
    text: str,
    candidate_names: tuple[str, ...],
    value: float,
) -> tuple[str, int]:
    total = 0

    for name in candidate_names:
        pattern = re.compile(
            rf"(?m)^(?P<prefix>\s*(?:export\s+)?{re.escape(name)}\s*=\s*)"
            rf"(?P<value>\"[^\"]*\"|'[^']*'|[-+]?[0-9]*\.?[0-9]+)"
            rf"(?P<suffix>\s*(?:#.*)?)$"
        )

        text, count = pattern.subn(
            lambda m: f"{m.group('prefix')}{value}{m.group('suffix')}",
            text,
        )
        total += count

    return text, total


def replace_setting(
    text: str,
    *,
    option: str,
    value: float,
    variable_names: tuple[str, ...],
) -> tuple[str, str]:
    changed, count = replace_cli_option(text, option, value)

    if count:
        return changed, f"CLI option ({count} replacement(s))"

    changed, count = replace_shell_assignment(
        text,
        variable_names,
        value,
    )

    if count:
        return changed, f"shell variable ({count} replacement(s))"

    names = ", ".join(variable_names)
    raise RuntimeError(
        f"Could not find {option!r} in '--option value' or '--option=value' form, "
        f"and could not find a simple assignment for: {names}.\n\n"
        "Please run:\n"
        "  grep -nEi 'cross.*mi|mi.*loss|align' "
        "run_UCF_experiment_train_and_checkpoint_256.sh\n"
        "and send me that output."
    )


def build_experiment_script(
    original: str,
    *,
    cross_mi: float,
    z_alignment: float,
) -> tuple[str, str, str]:
    text, mi_method = replace_setting(
        original,
        option="--cross_mi_loss_scale",
        value=cross_mi,
        variable_names=(
            "CROSS_MI_LOSS_SCALE",
            "CROSS_MI_SCALE",
            "CROSS_MI",
        ),
    )

    text, align_method = replace_setting(
        text,
        option="--z_alignment_loss_scale",
        value=z_alignment,
        variable_names=(
            "Z_ALIGNMENT_LOSS_SCALE",
            "Z_ALIGN_LOSS_SCALE",
            "Z_ALIGNMENT_SCALE",
            "Z_ALIGN_SCALE",
            "Z_ALIGNMENT",
            "Z_ALIGN",
        ),
    )

    return text, mi_method, align_method


def run_experiment(
    shell_script: Path,
    experiment: dict,
    *,
    dry_run: bool,
) -> None:
    original = shell_script.read_text(encoding="utf-8")

    modified, mi_method, align_method = build_experiment_script(
        original,
        cross_mi=experiment["cross_mi"],
        z_alignment=experiment["z_alignment"],
    )

    print("\n" + "=" * 76)
    print(f"Experiment          : {experiment['name']}")
    print(f"Cross-MI            : {experiment['cross_mi']}")
    print(f"Z alignment         : {experiment['z_alignment']}")
    print(f"Cross-MI replacement: {mi_method}")
    print(f"Z-align replacement : {align_method}")
    print(f"Source shell script : {shell_script}")
    print("=" * 76)

    if dry_run:
        print("DRY RUN: no training started.")
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".sh",
        prefix=f".tmp_{experiment['name']}_",
        dir=shell_script.parent,
        delete=False,
    ) as handle:
        handle.write(modified)
        temp_script = Path(handle.name)

    temp_script.chmod(0o755)

    try:
        subprocess.run(
            ["bash", str(temp_script)],
            cwd=shell_script.parent,
            check=True,
        )
    finally:
        temp_script.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--script",
        type=Path,
        default=Path(
            "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/src/commands/"
            "run_UCF_experiment_train_and_checkpoint_256.sh"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check replacements without starting training.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shell_script = args.script.expanduser().resolve()

    if not shell_script.is_file():
        print(f"ERROR: shell script not found: {shell_script}", file=sys.stderr)
        return 1

    try:
        for experiment in EXPERIMENTS:
            run_experiment(
                shell_script,
                experiment,
                dry_run=args.dry_run,
            )
    except subprocess.CalledProcessError as exc:
        print(
            f"\nERROR: experiment failed with exit code {exc.returncode}. "
            "The following experiment was not started.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print("\nBoth experiments completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())