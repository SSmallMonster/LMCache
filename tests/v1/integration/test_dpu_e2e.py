import asyncio
import pytest
import torch
from unittest.mock import Mock, patch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends


@pytest.fixture
def e2e_config():
    """Create end-to-end test configuration."""
    config = Mock(spec=LMCacheEngineConfig)
    config.enable_pd = False
    config.local_cpu = True
    config.max_local_cpu_size = 10 * 1024 * 1024  # 10MB
    config.enable_p2p = False
    config.local_disk = False
    config.gds_path = None
    config.maru_path = None
    config.remote_storage_plugins = None
    config.remote_url = None

    config.storage_plugins = ["dpu"]
    config.extra_config = {
        "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
        "storage_plugin.dpu.class_name": "DPUStorageBackend",
    }
    return config


class TestDPUEndToEnd:
    """End-to-end tests for DPU backend integration."""

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_complete_store_retrieve_cycle(self, mock_wrapper_class, e2e_config):
        """Test complete store and retrieve cycle through LMCache system."""
        # Setup mock DPU agent
        mock_wrapper = Mock()
        mock_wrapper.store_kv_cache.return_value = True

        # Mock retrieve to return test tensors
        test_k = torch.randn(4, 8, 64, dtype=torch.float16)
        test_v = torch.randn(4, 8, 64, dtype=torch.float16)
        mock_wrapper.retrieve_kv_cache.return_value = (test_k, test_v)
        mock_wrapper.contains_key.return_value = True

        mock_wrapper_class.return_value = mock_wrapper

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            # Create storage backends through LMCache system
            storage_backends = CreateStorageBackends(
                config=e2e_config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda:0",
            )

            # Get DPU backend
            assert "dpu" in storage_backends
            dpu_backend = storage_backends["dpu"]

            # Create test data
            key = CacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=12345,
                dtype=torch.float16,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            allocator = AdHocMemoryAllocator(device=device)

            # Create KV tensors in the expected format [2, layers, heads, seq_len, head_dim]
            kv_tensor = torch.stack([test_k, test_v], dim=0)  # [2, 4, 8, 64]

            mem_obj = allocator.allocate(
                kv_tensor.shape,
                torch.float16,
                fmt=MemoryFormat.KV_2LTD
            )
            mem_obj.tensor.copy_(kv_tensor)

            # Test complete workflow
            result = dpu_backend.batched_submit_put_task([key], [mem_obj])
            assert result is None  # Synchronous operation

            # Verify store was called
            mock_wrapper.store_kv_cache.assert_called_once()

            # Test contains operation
            assert dpu_backend.contains(key) is True

            # Test retrieve operation
            retrieved_obj = dpu_backend.get_blocking(key)
            assert retrieved_obj is not None

            # Verify retrieve was called
            mock_wrapper.retrieve_kv_cache.assert_called_once()

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_fallback_mechanism_e2e(self, mock_wrapper_class, e2e_config):
        """Test fallback mechanism in end-to-end scenario."""
        # Setup mock to fail on first store
        mock_wrapper = Mock()
        mock_wrapper.store_kv_cache.side_effect = Exception("DPU connection failed")
        mock_wrapper.contains_key.return_value = False  # Not in DPU
        mock_wrapper_class.return_value = mock_wrapper

        # Enable fallback in config
        e2e_config.extra_config["fallback_enabled"] = True

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=e2e_config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda:0",
            )

            dpu_backend = storage_backends["dpu"]

            # Create test data
            key = CacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=54321,
                dtype=torch.float16,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            allocator = AdHocMemoryAllocator(device=device)
            mem_obj = allocator.allocate(
                torch.Size([2, 4, 8, 64]),
                torch.float16,
                fmt=MemoryFormat.KV_2LTD
            )

            # Store should use fallback due to DPU failure
            result = dpu_backend.batched_submit_put_task([key], [mem_obj])
            assert result is None

            # Backend should have switched to fallback mode
            assert dpu_backend.dpu_available is False
            assert str(key) in dpu_backend.fallback_storage

            # Contains should work via fallback
            assert dpu_backend.contains(key) is True

            # Retrieve should work via fallback
            retrieved_obj = dpu_backend.get_blocking(key)
            assert retrieved_obj is not None
            assert retrieved_obj == mem_obj  # Should be same object from fallback storage