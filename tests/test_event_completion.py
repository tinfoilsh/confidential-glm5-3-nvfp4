import unittest

import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler, Transfer

BLOCK_BYTES = 1024
BLOCK_COUNT = 4
PATTERN = 73
PENDING_CYCLES = 500_000_000
JOB_ID = 123


class EventCompletionTests(unittest.TestCase):
    def setUp(self):
        self.cpu = torch.full((BLOCK_COUNT, BLOCK_BYTES), PATTERN, dtype=torch.int8)
        self.gpu = torch.zeros_like(self.cpu, device='cuda')
        self.handler = SingleDirectionOffloadingHandler(
            [self.gpu],
            [self.cpu],
            1,
            [[CanonicalKVCacheRef(0, BLOCK_BYTES)]],
            False,
            host_memory_is_pinned=False,
        )

    def tearDown(self):
        torch.cuda.synchronize()
        self.handler.shutdown()

    def test_completed_copy_preserves_data_and_recycles_once(self):
        blocks = list(range(BLOCK_COUNT))
        self.handler.transfer_async(
            JOB_ID,
            CPULoadStoreSpec(blocks),
            GPULoadStoreSpec(blocks, [BLOCK_COUNT], [0]),
        )
        self.handler.wait({JOB_ID})
        results = self.handler.get_finished()
        self.assertEqual([r.job_id for r in results], [JOB_ID])
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].transfer_size, BLOCK_BYTES * BLOCK_COUNT)
        self.assertGreaterEqual(results[0].transfer_time, 0)
        self.assertTrue(torch.equal(self.gpu.cpu(), self.cpu))
        self.assertEqual(self.handler.get_finished(), [])
        self.assertEqual(len(self.handler._event_pool), 2)
        self.assertNotIn(JOB_ID, self.handler._transfer_events)

    def test_incomplete_start_does_not_release_transfer(self):
        # Separate streams make unequal event readiness deterministic.
        slow = torch.cuda.Stream()
        fast = torch.cuda.Stream()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        with torch.cuda.stream(slow):
            torch.cuda._sleep(PENDING_CYCLES)
            start.record(slow)
        end.record(fast)
        end.synchronize()
        descriptors = torch.empty(0, dtype=torch.int64)
        transfer = Transfer(
            JOB_ID, slow, start, end, 0, descriptors, descriptors, descriptors,
        )
        self.handler._transfers.append(transfer)
        self.handler._transfer_events[JOB_ID] = end
        self.assertFalse(start.query())
        self.assertEqual(self.handler.get_finished(), [])
        self.assertIs(self.handler._transfers[0], transfer)
        self.assertEqual(self.handler._event_pool, [])
        self.assertIn(JOB_ID, self.handler._transfer_events)


if __name__ == '__main__':
    unittest.main()
