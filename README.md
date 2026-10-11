# confidential-glm5-3-nvfp4

vLLM 0.31.0 image for GLM-5.3 (Inferact NVFP4) on 4x NVIDIA B300 with
confidential computing and a 1,048,576-token context window.

Each enclave serves one TP4 replica. Deploy two enclaves to use all eight GPUs
on a B300 node; request routing remains outside this model container.

The serving configuration uses the V2 model runner, asynchronous scheduling,
MTP5, FP8 KV, and opt-in FP32 target routing with K256 tiles. The draft router
retains its upstream precision. Native CPU prefix caching retains reusable
prompt KV after GPU eviction. See [tinfoil-config.yml](tinfoil-config.yml) for
the pinned model revision, CVM version, resource allocation, and serving flags.

## Build

The [Dockerfile](Dockerfile) applies the [vLLM patches](patches/) to a
digest-pinned base image and packages the inference sidecar.

[build/](build/) prepares FlashInfer for offline startup: it downloads missing
B300 SM103a kernels, verifies artifact checksums, and creates export-header
links before the container runs with a read-only root filesystem.

```sh
docker build -t confidential-glm5-3-nvfp4 .
```

CC patch documentation, regression tests, and qualification status are maintained
in the [private CC lab](https://github.com/tinfoilsh/glm-b300-cc-lab/tree/main/validation/full-model-runtime).
