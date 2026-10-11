# syntax=docker/dockerfile:1.6
#
# Patched vLLM image for GLM-5.3 NVFP4 (Inferact) on 4x NVIDIA B300.
# Base is digest-pinned for attestation. See patches/ for the diff set.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
ARG SIDECAR_IMAGE=ghcr.io/tinfoilsh/inference-sidecar@sha256:65ce23d6560c46a1e8614ede187fcbf9798b267aa33878905b4872404787f47d
FROM ${SIDECAR_IMAGE} AS sidecar

FROM ${VLLM_BASE_IMAGE}

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    test -x /usr/bin/patch; \
    cd /; \
    sha256sum --check /tmp/tinfoil-patches/before.sha256; \
    for p in /tmp/tinfoil-patches/*.patch; do \
        echo "Applying $(basename "$p")"; \
        /usr/bin/patch -p1 --no-backup-if-mismatch --fuzz=0 < "$p"; \
    done; \
    sha256sum --check /tmp/tinfoil-patches/after.sha256; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true; \
    rm -rf /tmp/tinfoil-patches; \
    python3 -c "import vllm; print('vllm', vllm.__version__, 'with tinfoil patches')"

# Bake FlashInfer cubins at build time: the enclave has no egress for JIT
# downloads and the container rootfs is read-only, so the symlinks
# ensure_symlink() would create at runtime are pre-created here.
COPY build/ /tmp/tinfoil-build/
RUN mkdir -p /opt/tinfoil && \
    FLASHINFER_CUDA_ARCH_LIST=10.3a python3 /tmp/tinfoil-build/bake-artifacts.py && \
    rm -rf /tmp/tinfoil-build

COPY --from=sidecar /inference-sidecar /opt/tinfoil/inference-sidecar
ENTRYPOINT ["/opt/tinfoil/inference-sidecar", "vllm", "serve"]
