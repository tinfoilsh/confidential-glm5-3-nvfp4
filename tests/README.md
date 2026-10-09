Run the CPU checks inside the built image. The virtual environment reuses its
matching vLLM and PyTorch installation:

```sh
uv venv --system-site-packages /tmp/cc-tests
env -u UV_OVERRIDE uv pip install --python /tmp/cc-tests/bin/python \
  --require-hashes -r tests/requirements.txt
/tmp/cc-tests/bin/python -m pytest --noconftest -q tests/test_cc_runtime.py
```

These checks exercise the UVA override, buffer copies and reuse, and the router
tile selection boundary. They do not load model weights or validate GPU kernels.
