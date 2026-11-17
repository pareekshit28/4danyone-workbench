FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv PYTHONUNBUFFERED=1 \
    WORKBENCH_DATA=/workspace/4danyone-workbench HF_HOME=/workspace/4danyone-workbench/hf \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ffmpeg git libgl1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*
RUN uv venv --seed --python 3.11 /opt/venv
WORKDIR /opt/4danyone
COPY requirements.txt requirements-4dgs.txt ./
COPY workbench/requirements.txt workbench/requirements.txt
RUN uv pip install --index https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 \
    && uv pip install setuptools wheel packaging ninja \
    && sed '/^gsplat @ /d' requirements-4dgs.txt > /tmp/reconstruction.txt \
    && uv pip install -r requirements.txt -r /tmp/reconstruction.txt -r workbench/requirements.txt \
    && uv cache clean
# Ampere, Ada, Hopper and Blackwell. PTX covers newer compatible devices.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
RUN MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    python -m pip install --no-cache-dir --no-build-isolation \
    "$(sed -n 's/^gsplat @ //p' requirements-4dgs.txt)"
COPY fdanyone fdanyone
# Use the upstream loader to build once, then install the binary for direct import.
RUN MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" TORCH_EXTENSIONS_DIR=/tmp/torch-build \
    python -c 'from fdanyone.reconstruction.kernels import extension; print(extension().__file__)' \
    && cp /tmp/torch-build/fdanyone_4dgs_kernels/fdanyone_4dgs_kernels.so fdanyone/reconstruction/ \
    && rm -rf /tmp/torch-build
# 4DAnyone records the GVHMR commit in its motion cache. This checkout must
# own its Git metadata; a Docker copy of a submodule loses its parent gitdir.
ARG GVHMR_REVISION=6ec3ca39336c50492c0fae65fba2fb831fc7d866
RUN git init third_party/GVHMR \
    && git -C third_party/GVHMR fetch --depth=1 https://github.com/zju3dv/GVHMR.git "$GVHMR_REVISION" \
    && git -C third_party/GVHMR checkout --detach FETCH_HEAD \
    && test "$(git -C third_party/GVHMR rev-parse HEAD)" = "$GVHMR_REVISION" \
    && test "$(git -C third_party/GVHMR rev-parse --show-toplevel)" = "$PWD/third_party/GVHMR" \
    && test -f third_party/GVHMR/hmr4d/__init__.py \
    && test -f third_party/GVHMR/tools/demo/demo.py
COPY inference.py reconstruct.py ./
COPY scripts scripts
COPY workbench workbench
RUN python -m pip check && python -c 'import torch; import gsplat.csrc; from fdanyone.reconstruction.kernels import extension; extension()' \
    && python inference.py --help >/dev/null && python reconstruct.py --help >/dev/null \
    && python -c 'from workbench.server import app'
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["sh", "workbench/start.sh"]
