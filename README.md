# confidential-glm5-3-nvfp4

vLLM image for GLM-5.3 (Inferact NVFP4) on 8x NVIDIA Blackwell.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3?hardware=b300&variant=nvfp4)
(nvfp4 variant, B300). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `Inferact/GLM-5.3-NVFP4@ce67b36f` and served from a
  verified model pack (pack schema 2).
- `--attention-backend FLASHINFER_MLA_SPARSE` pinned (measured best on SM100
  for this family).
- FlashInfer cubins baked at build time (the container runs offline).
- FlashInfer autotuning disabled (`--no-enable-flashinfer-autotune`); the
  image lacks the header its timing kernel needs, so it would fall back to
  event timing anyway.
- Patches in `patches/`, one line each in the header of the patch file.
