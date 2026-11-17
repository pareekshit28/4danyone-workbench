"""Two small CUDA kernels, compiled lazily; no reconstruction framework dependency."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def extension():
    try:
        from . import fdanyone_4dgs_kernels
    except ImportError as exc:
        if Path(__file__).with_name("fdanyone_4dgs_kernels.so").exists():
            raise RuntimeError("The prebuilt 4DGS CUDA extension could not be loaded.") from exc
    else:
        return fdanyone_4dgs_kernels

    from torch.utils.cpp_extension import load

    root = Path(__file__).parent / "cuda"
    # load() hashes only the listed sources; include header contents in the build flags.
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(root.glob("*.cu*")))).hexdigest()[:16]
    return load(
        name="fdanyone_4dgs_kernels",
        sources=[str(root / "extension.cu")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", f"-DFDANYONE_KERNEL_SOURCE={digest}"],
    )


class Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, left, right):
        inputs = tuple(t.contiguous() for t in (scale, left, right))
        ctx.save_for_backward(*inputs)
        return tuple(extension().covariance(*inputs))

    @staticmethod
    def backward(ctx, covariance, velocity, variance):
        gradients = [g.contiguous() for g in (covariance, velocity, variance)]
        return tuple(extension().covariance_backward(*ctx.saved_tensors, *gradients))
