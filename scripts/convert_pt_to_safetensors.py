#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint to safetensors for Candle."
    )
    parser.add_argument("src", type=Path, help="Source .pt checkpoint path")
    parser.add_argument("dst", type=Path, help="Destination .safetensors path")
    return parser.parse_args()


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint does not contain a tensor dictionary")

    return {
        key: value.detach().clone().contiguous()
        for key, value in checkpoint.items()
        if hasattr(value, "shape")
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.src, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)

    if (
        "_encoder.shared.weight" in state_dict
        and "_encoder.encoder.embed_tokens.weight" in state_dict
    ):
        del state_dict["_encoder.shared.weight"]

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, args.dst)

    print("wrote", args.dst)
    print("num tensors:", len(state_dict))


if __name__ == "__main__":
    main()
