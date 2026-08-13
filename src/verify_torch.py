"""Verify that PyTorch detects and can compute on the AMD ROCm GPU."""

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/ROCm device available to PyTorch")

    device = torch.device("cuda")

    print("Torch version:    ", torch.__version__)
    print("Device:           ", torch.cuda.get_device_name(0))
    print("Device count:     ", torch.cuda.device_count())

    x = torch.ones((1024, 1024), dtype=torch.float32, device=device)
    y = torch.arange(1024 * 1024, dtype=torch.float32, device=device).reshape(1024, 1024)

    z = (x @ y).sum()
    torch.cuda.synchronize()

    print("Matmul sum (GPU): ", float(z))
    print("Torch verification OK")


if __name__ == "__main__":
    main()
