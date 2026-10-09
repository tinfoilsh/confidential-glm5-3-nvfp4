# confidential-glm5-3-nvfp4

vLLM 0.31.0 image for GLM-5.3 (Inferact NVFP4) on 8x NVIDIA B300 with
confidential computing and a 1,048,576-token context window.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3?hardware=b300&variant=nvfp4)
(nvfp4 variant, B300). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `Inferact/GLM-5.3-NVFP4@ce67b36f` and served from a
  verified model pack (pack schema 2).
- `--attention-backend FLASHINFER_MLA_SPARSE` pinned (measured best on SM100
  for this family).
- FlashInfer cubins baked at build time (the container runs offline).
- FlashInfer autotuning disabled (`--no-enable-flashinfer-autotune`) to match
  the measured candidate.
- Patches in `patches/`, one line each in the header of the patch file.

The serving configuration uses the V2 model runner, asynchronous scheduling,
MTP5, FP8 KV, and opt-in FP32 target routing with K256 tiles. The draft router
retains its upstream precision. See [patches](patches/README.md) for the CC
copy fallback, MTP head-loading repair, and router changes.

The measured configuration allocates 32 vCPUs and 768 GiB of guest RAM, with
a 384 GiB shared-memory ceiling and 2 GiB Run:ai loader buffers. Build-time
artifact verification includes the B300 SM103a FlashInfer objects and export
links needed for offline serving. The existing inference sidecar handles usage;
the image does not install the lab's worker extension or enable development RPCs.

The selected inference code passed concurrent serving checks and the coding
workload screen. Some requests exhausted the benchmark's 8,192-token output cap,
and near-1M recall has varied across runs. These results do not establish general
quality equivalence; the benchmark cap is not a server setting in this recipe.
