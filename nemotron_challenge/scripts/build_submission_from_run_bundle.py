"""Build a Kaggle submission.zip from a Colab run bundle.

The run bundle is the portable artifact downloaded from Colab. It must contain:

    adapter/adapter_config.json
    adapter/adapter_model.safetensors

This script writes the Kaggle artifact with those two files at the zip root.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


BUNDLE_ADAPTER_CONFIG = "adapter/adapter_config.json"
BUNDLE_ADAPTER_WEIGHTS = "adapter/adapter_model.safetensors"


def normalized_zip_names(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map normalized POSIX-style names to original archive names."""
    return {name.replace("\\", "/"): name for name in archive.namelist()}


def validate_bundle(bundle_zip: Path) -> tuple[str, str]:
    if not bundle_zip.is_file():
        raise FileNotFoundError(f"run bundle not found: {bundle_zip}")

    with zipfile.ZipFile(bundle_zip) as archive:
        names = normalized_zip_names(archive)
        missing = [
            name
            for name in (BUNDLE_ADAPTER_CONFIG, BUNDLE_ADAPTER_WEIGHTS)
            if name not in names
        ]
        if missing:
            raise FileNotFoundError(f"run bundle is missing required files: {missing}")

        config = json.loads(archive.read(names[BUNDLE_ADAPTER_CONFIG]).decode("utf-8"))
        rank = int(config.get("r", 0))
        if not 1 <= rank <= 32:
            raise ValueError(f"Kaggle submissions require LoRA rank <= 32; found r={rank}")
        print("adapter rank:", rank)

        return names[BUNDLE_ADAPTER_CONFIG], names[BUNDLE_ADAPTER_WEIGHTS]


def build_submission(bundle_zip: Path, output_zip: Path) -> None:
    config_name, weights_name = validate_bundle(bundle_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(bundle_zip) as source:
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            target.writestr("adapter_config.json", source.read(config_name))
            target.writestr("adapter_model.safetensors", source.read(weights_name))

    with zipfile.ZipFile(output_zip) as archive:
        names = sorted(archive.namelist())
    expected = ["adapter_config.json", "adapter_model.safetensors"]
    if names != expected:
        raise RuntimeError(f"unexpected submission contents: {names}")

    print("wrote:", output_zip)
    print("zip contents:", names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_bundle_zip", type=Path, help="Path to *_run_bundle.zip")
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="Output Kaggle submission zip. Defaults beside the bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_zip = args.run_bundle_zip.resolve()
    output_zip = args.output_zip
    if output_zip is None:
        output_zip = bundle_zip.with_name(bundle_zip.stem.replace("_run_bundle", "") + "_submission.zip")
    build_submission(bundle_zip, output_zip.resolve())


if __name__ == "__main__":
    main()
