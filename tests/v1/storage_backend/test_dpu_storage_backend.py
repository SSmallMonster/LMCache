# tests/v1/storage_backend/test_dpu_storage_backend.py
import asyncio
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.dpu_storage_backend import DPUStorageBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@pytest.fixture
def mock_config():
    """Create mock LMCache configuration."""
    config = Mock(spec=LMCacheEngineConfig)
    config.storage_backend = Mock()
    config.storage_backend.config = {
        "dpu_device_pci": "03:00.0",
        "max_concurrent_ops": 16,
        "connection_timeout": 5.0,
        "retry_attempts": 3,
        "fallback_enabled": True,
        "metadata_cache_size": 1000,
    }
    return config


@pytest.fixture
def mock_metadata():
    """Create mock metadata."""
    metadata = Mock(spec=LMCacheMetadata)
    metadata.role = "worker"
    return metadata


@pytest.fixture
def mock_local_cpu_backend():
    """Create mock local CPU backend."""
    backend = Mock(spec=LocalCPUBackend)
    allocator = Mock()
    backend.get_memory_allocator.return_value = allocator
    return backend


@pytest.fixture
def test_key():
    """Create test cache engine key."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.float16,
    )


@pytest.fixture
def test_memory_obj():
    """Create test memory object."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allocator = AdHocMemoryAllocator(device=device)
    mem_obj = allocator.allocate(
        torch.Size([4, 32, 128]),
        torch.float16,
        fmt=MemoryFormat.KV_2LTD
    )
    # Fill with test data
    mem_obj.tensor.fill_(0.5)
    return mem_obj


class TestDPUStorageBackend:

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_init_success(self, mock_wrapper_class, mock_config, mock_metadata, mock_local_cpu_backend):
        """Test successful DPU storage backend initialization."""
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
            loop=asyncio.get_event_loop()
        )

        assert backend.dpu_agent == mock_wrapper
        assert backend.local_cpu_backend == mock_local_cpu_backend
        assert backend.dpu_available is True
        assert backend.fallback_storage == {}

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_contains_key_exists(self, mock_wrapper_class, mock_config, mock_metadata,
                                mock_local_cpu_backend, test_key):
        """Test contains method when key exists."""
        mock_wrapper = Mock()
        mock_wrapper.contains_key.return_value = True
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        result = backend.contains(test_key)

        assert result is True
        mock_wrapper.contains_key.assert_called_once_with(str(test_key))

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_contains_key_not_exists(self, mock_wrapper_class, mock_config, mock_metadata,
                                    mock_local_cpu_backend, test_key):
        """Test contains method when key does not exist."""
        mock_wrapper = Mock()
        mock_wrapper.contains_key.return_value = False
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        result = backend.contains(test_key)

        assert result is False

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_batched_submit_put_task_success(self, mock_wrapper_class, mock_config,
                                           mock_metadata, mock_local_cpu_backend,
                                           test_key, test_memory_obj):
        """Test successful batched put operation."""
        mock_wrapper = Mock()
        mock_wrapper.store_kv_cache.return_value = True
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        keys = [test_key]
        objs = [test_memory_obj]

        result = backend.batched_submit_put_task(keys, objs)

        assert result is None  # Synchronous operation returns None
        mock_wrapper.store_kv_cache.assert_called_once()

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_get_blocking_success(self, mock_wrapper_class, mock_config, mock_metadata,
                                 mock_local_cpu_backend, test_key):
        """Test successful blocking get operation."""
        # Setup mock wrapper
        mock_wrapper = Mock()
        k_tensor = torch.randn(4, 32, 128, dtype=torch.float16)
        v_tensor = torch.randn(4, 32, 128, dtype=torch.float16)
        mock_wrapper.retrieve_kv_cache.return_value = (k_tensor, v_tensor)
        mock_wrapper_class.return_value = mock_wrapper

        # Setup mock allocator
        mock_allocator = Mock()
        mock_mem_obj = Mock(spec=MemoryObj)
        mock_mem_obj.tensor = torch.cat([k_tensor.flatten(), v_tensor.flatten()])
        mock_allocator.allocate.return_value = mock_mem_obj
        mock_local_cpu_backend.get_memory_allocator.return_value = mock_allocator

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        result = backend.get_blocking(test_key)

        assert result is not None
        mock_wrapper.retrieve_kv_cache.assert_called_once_with(str(test_key), "cuda")

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_get_blocking_not_found(self, mock_wrapper_class, mock_config, mock_metadata,
                                   mock_local_cpu_backend, test_key):
        """Test blocking get when key not found."""
        mock_wrapper = Mock()
        mock_wrapper.retrieve_kv_cache.return_value = None
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        result = backend.get_blocking(test_key)
        assert result is None

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_fallback_on_dpu_failure(self, mock_wrapper_class, mock_config, mock_metadata,
                                    mock_local_cpu_backend, test_key, test_memory_obj):
        """Test fallback to local storage on DPU failure."""
        mock_wrapper = Mock()
        mock_wrapper.store_kv_cache.side_effect = Exception("DPU connection failed")
        mock_wrapper_class.return_value = mock_wrapper

        # Enable fallback in config
        mock_config.storage_backend.config["fallback_enabled"] = True

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        keys = [test_key]
        objs = [test_memory_obj]

        # Should not raise exception, should fallback
        result = backend.batched_submit_put_task(keys, objs)

        # Should have fallen back to local storage
        assert backend.dpu_available is False
        assert str(test_key) in backend.fallback_storage

    def test_get_allocator_backend(self, mock_config, mock_metadata, mock_local_cpu_backend):
        """Test get_allocator_backend returns local CPU backend."""
        with patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper'):
            backend = DPUStorageBackend(
                dst_device="cuda",
                config=mock_config,
                metadata=mock_metadata,
                local_cpu_backend=mock_local_cpu_backend,
            )

            allocator = backend.get_allocator_backend()
            assert allocator == mock_local_cpu_backend

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_exists_in_put_tasks(self, mock_wrapper_class, mock_config, mock_metadata,
                                mock_local_cpu_backend, test_key):
        """Test exists_in_put_tasks method."""
        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        # For DPU backend, this should return False as we don't track put tasks
        result = backend.exists_in_put_tasks(test_key)
        assert result is False

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_pin_unpin_operations(self, mock_wrapper_class, mock_config, mock_metadata,
                                 mock_local_cpu_backend, test_key):
        """Test pin and unpin operations."""
        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        # Pin should return True (no-op for DPU backend)
        result = backend.pin(test_key)
        assert result is True

        # Unpin should return True (no-op for DPU backend)
        result = backend.unpin(test_key)
        assert result is True

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_remove_success(self, mock_wrapper_class, mock_config, mock_metadata,
                           mock_local_cpu_backend, test_key):
        """Test successful remove operation."""
        mock_wrapper = Mock()
        mock_wrapper.remove_key.return_value = True
        mock_wrapper_class.return_value = mock_wrapper

        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        result = backend.remove(test_key)

        assert result is True
        mock_wrapper.remove_key.assert_called_once_with(str(test_key))

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_close(self, mock_wrapper_class, mock_config, mock_metadata, mock_local_cpu_backend):
        """Test close operation."""
        backend = DPUStorageBackend(
            dst_device="cuda",
            config=mock_config,
            metadata=mock_metadata,
            local_cpu_backend=mock_local_cpu_backend,
        )

        # Close should not raise exception
        backend.close()
        # For now, close is a no-op, so we just verify it doesn't crash
        assert True