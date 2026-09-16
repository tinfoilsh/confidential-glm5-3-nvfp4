# vLLM patches

Unified-diff patches applied on top of the base image in `../Dockerfile`.
Applied in filename order at build time.

## Rules

- Diffs are `-p1` rooted at `/`, so paths start with
  `usr/local/lib/python3.12/dist-packages/...`
- Each patch is one reviewable change with a prefixed number
  (`NNNN-short-slug.patch`)
- Each patch includes an in-code comment citing the upstream issue/PR so
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

### vLLM v0.29.0 (diagnostic build, `v0.0.2-vllm029`)

| Patch | Class | Upstream ref | Dry-run vs v0.29.0 | Disposition |
|---|---|---|---|---|
| 0001-uva-device-mirror | CC-environment adaptation | (environment, not a vLLM bug) | APPLIES at fuzz=0; `buffer_utils.py` unchanged v0.28.0→v0.29.0 | keep |
| 0002-v2-spec-reject-invalid-drafts | upstream bug backport | vllm#46118, vllm#49694 (both still open) | APPLIES at fuzz=0; V2 runner still never reads `num_invalid_spec_tokens`; vllm#51450 merged after the v0.29.0 tag | keep |

Patched files pass pyflakes on the v0.29.0 source. Every recipe flag and
env var still exists in v0.29.0. `VLLM_ALLREDUCE_USE_FLASHINFER=0` is now
load-bearing (v0.29.0 flips FlashInfer all-reduce on by default, vllm#52998).
