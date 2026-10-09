# vLLM patches

Unified-diff patches applied on top of the base image in `../Dockerfile`.
Applied in filename order at build time.

## Rules

- Diffs are `-p1` rooted at `/`, so paths start with
  `usr/local/lib/python3.12/dist-packages/...`
- Each patch is one reviewable change with a prefixed number
  (`NNNN-short-slug.patch`)
- Each patch header cites the related upstream issue/PR so
  a future reader knows when it can be retired after a base-image bump

## Adding a patch

```bash
F=vllm/model_executor/layers/attention/mla_attention.py
BASE=$(sed -n 's/^ARG VLLM_BASE_IMAGE=//p' ../Dockerfile)

docker run --rm "$BASE" cat "/usr/local/lib/python3.12/dist-packages/$F" > orig
cp orig patched && $EDITOR patched
diff -u --label "a/usr/local/lib/python3.12/dist-packages/$F" \
        --label "b/usr/local/lib/python3.12/dist-packages/$F" \
        orig patched > NNNN-slug.patch
docker build ..
```

## Triage log

### vLLM v0.31.0

Pinned upstream commit: `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
The build verifies every affected source file against `before.sha256`, applies
the patches without fuzz, and verifies the result against `after.sha256`.

| Patch | Purpose |
|---|---|
| 0001-explicit-uva-fallback | Registers `VLLM_DISABLE_UVA=1` and selects upstream explicit-copy buffers when host memory cannot provide a live device alias. Pinned-memory support remains unchanged. Related: [#57404](https://github.com/vllm-project/vllm/pull/57404). |
| 0002-defer-full-glm-mtp-head | Defers the disposable MTP output head for full GLM with V2 and pipeline parallelism one; the existing loader shares the loaded target head. Preserves the quantization NaN check. Related: [#59306](https://github.com/vllm-project/vllm/issues/59306). |
| 0003-opt-in-fp32-router | Enables FP32 target gate weights and the existing batch-invariant linear kernel through `moe_router_batch_invariant`. This is a component control, not global model invariance. Related: [#59985](https://github.com/vllm-project/vllm/pull/59985). |
| 0004-small-batch-router-k256 | Tunes that FP32 kernel on Blackwell for shape `(N,K)=(256,6144)` and at most 256 rows; other shapes, dtypes, and larger batches retain upstream selection. |

The legacy invalid-draft patch is superseded by upstream structured-output work
in [#48200](https://github.com/vllm-project/vllm/pull/48200) and
[#58612](https://github.com/vllm-project/vllm/pull/58612). The old device-mirror
patch is replaced by the explicit UVA gate above. Neither legacy patch applies
to this build.

The deployed configuration pins TRTLLM collectives and preserves MTP5, FP8 KV,
and 1M context. It includes no repetition-detection default override, output-token
cap, experimental transfer policy, or local draft-argmax change.

### vLLM v0.29.0 (diagnostic build, `v0.0.2-vllm029`)

| Patch | Class | Upstream ref | Dry-run vs v0.29.0 | Disposition |
|---|---|---|---|---|
| 0001-uva-device-mirror | CC-environment adaptation | (environment, not a vLLM bug) | APPLIES at fuzz=0; `buffer_utils.py` unchanged v0.28.0→v0.29.0 | keep |
| 0002-v2-spec-reject-invalid-drafts | upstream bug backport | vllm#46118, vllm#49694 (both still open) | APPLIES at fuzz=0; V2 runner still never reads `num_invalid_spec_tokens`; vllm#51450 merged after the v0.29.0 tag | keep |

Patched files pass pyflakes on the v0.29.0 source. Every recipe flag and
env var still exists in v0.29.0. `VLLM_ALLREDUCE_USE_FLASHINFER=0` is now
load-bearing (v0.29.0 flips FlashInfer all-reduce on by default, vllm#52998).
