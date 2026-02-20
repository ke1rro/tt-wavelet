import argparse
import tomllib
from os import PathLike
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(config_path: str = "config.toml", output_path: PathLike = "data/input"):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for input_descriptor in tqdm(config["inputs"], desc="Generating signals"):
        name = input_descriptor["name"]
        shape = input_descriptor["shape"]
        magnitude = input_descriptor["magnitude"]

        signal = np.random.rand(*shape) * magnitude
        with open(output_path / name, "wb") as f:
            f.write(signal.astype("<f8").tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate input signals for wavelet transform precision experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/input",
        help="Directory to save generated signals.",
    )
    args = parser.parse_args()

    main(config_path=args.config, output_path=args.output)
