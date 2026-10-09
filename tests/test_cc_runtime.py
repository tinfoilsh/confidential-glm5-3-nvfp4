import pytest
import torch

from vllm.model_executor.determinism import batch_invariant_configs as configs
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
from vllm.v1.worker.gpu import buffer_utils

ROUTER_EXPERTS = 256
ROUTER_HIDDEN_SIZE = 6144
ROUTER_ROW_LIMIT = 256
EXPECTED_ROUTER_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}


@pytest.fixture
def reset_uva_cache():
    is_pin_memory_available.cache_clear()
    is_uva_available.cache_clear()
    yield
    is_pin_memory_available.cache_clear()
    is_uva_available.cache_clear()


@pytest.mark.parametrize("disable_uva", [None, "0", "1"])
def test_uva_override_preserves_pinned_memory(
    monkeypatch, reset_uva_cache, disable_uva
):
    monkeypatch.setattr(current_platform, "is_pin_memory_available", lambda: True)
    if disable_uva is None:
        monkeypatch.delenv("VLLM_DISABLE_UVA", raising=False)
    else:
        monkeypatch.setenv("VLLM_DISABLE_UVA", disable_uva)
    assert is_pin_memory_available()
    assert is_uva_available() == (disable_uva != "1")


def test_copy_fallback_owns_data_and_refreshes_reused_buffers(
    monkeypatch, reset_uva_cache
):
    monkeypatch.setenv("VLLM_DISABLE_UVA", "1")
    monkeypatch.setattr(current_platform, "device_type", "cpu")

    def reject_alias(*args, **kwargs):
        pytest.fail("Disabled UVA must not create a host/device alias")

    monkeypatch.setattr(
        buffer_utils, "get_accelerator_view_from_cpu_tensor", reject_alias
    )
    initial_capacity = 2
    with pytest.raises(RuntimeError, match="UVA is not available"):
        buffer_utils.UvaBuffer(initial_capacity, torch.int32)
    pool = buffer_utils.UvaBufferPool(
        initial_capacity, torch.int32, max_concurrency=2
    )
    for length in (3, 7, 2, 0):
        values = torch.arange(length, dtype=torch.int32)
        copied = pool.copy_to_uva(values)
        values.fill_(-1)
        assert copied.tolist() == list(range(length))
    backed = buffer_utils.UvaBackedTensor(3, torch.int32)
    backed.np[:] = [4, 5, 6]
    assert backed.copy_to_uva().tolist() == [4, 5, 6]
    staged = buffer_utils.StagedWriteTensor(
        (2, 8), torch.int32, torch.device("cpu"), uva_instead_of_gpu=True
    )
    assert staged.gpu.device.type == "cpu"
    assert staged.write_contents is None


@pytest.fixture
def blackwell_config(monkeypatch):
    monkeypatch.setattr(configs, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    monkeypatch.setattr(
        configs,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        configs._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["blackwell"],
    )


@pytest.mark.parametrize("rows", [1, 16, ROUTER_ROW_LIMIT])
def test_small_fp32_router_selects_k256(blackwell_config, rows):
    selected = configs._get_matmul_config(
        rows, ROUTER_EXPERTS, ROUTER_HIDDEN_SIZE, torch.float32, {}
    )
    assert selected == EXPECTED_ROUTER_CONFIG


@pytest.mark.parametrize(
    "rows,experts,hidden,dtype",
    [
        (ROUTER_ROW_LIMIT + 1, ROUTER_EXPERTS, ROUTER_HIDDEN_SIZE, torch.float32),
        (1, ROUTER_EXPERTS // 2, ROUTER_HIDDEN_SIZE, torch.float32),
        (1, ROUTER_EXPERTS, ROUTER_HIDDEN_SIZE // 2, torch.float32),
        (1, ROUTER_EXPERTS, ROUTER_HIDDEN_SIZE, torch.float16),
    ],
)
def test_other_router_inputs_preserve_default(
    blackwell_config, rows, experts, hidden, dtype
):
    default = {"BLOCK_SIZE_K": 32}
    assert configs._get_matmul_config(rows, experts, hidden, dtype, default) is default


def test_non_blackwell_preserves_default(monkeypatch, blackwell_config):
    monkeypatch.setattr(configs, "_TUNED_MATMUL_CONFIGS_FOR_DEVICE", None)
    default = {"BLOCK_SIZE_K": 32}
    assert configs._get_matmul_config(
        1, ROUTER_EXPERTS, ROUTER_HIDDEN_SIZE, torch.float32, default
    ) is default
