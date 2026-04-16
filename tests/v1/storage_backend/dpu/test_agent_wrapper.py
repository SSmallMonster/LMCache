# tests/v1/storage_backend/dpu/test_agent_wrapper.py
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from lmcache.v1.storage_backend.dpu.agent_wrapper import DPUAgentWrapper
from lmcache.v1.storage_backend.dpu.config import DPUConfig
from lmcache.v1.storage_backend.dpu.exceptions import (
    DPUConnectionError,
    DPUOperationError,
)


@pytest.fixture
def dpu_config():
    """Create test DPU configuration."""
    return DPUConfig.from_dict({
        "dpu_device_pci": "03:00.0",
        "max_concurrent_ops": 16,
        "connection_timeout": 5.0,
        "retry_attempts": 3,
        "fallback_enabled": True,
        "metadata_cache_size": 1000,
    })


@pytest.fixture
def mock_dpu_agent():
    """Mock DPU agent for testing."""
    with patch('lmcache.v1.storage_backend.dpu.agent_wrapper.dpu_agent') as mock_agent:
        mock_instance = MagicMock()
        mock_agent.return_value = mock_instance
        yield mock_agent, mock_instance


class TestDPUAgentWrapper:
    def test_init_success(self, dpu_config, mock_dpu_agent):
        """Test successful DPU agent wrapper initialization."""
        mock_agent_class, mock_instance = mock_dpu_agent

        wrapper = DPUAgentWrapper(dpu_config)

        # Verify agent was created with correct config
        mock_agent_class.assert_called_once()
        assert wrapper.config == dpu_config
        assert wrapper.agent == mock_instance
        assert wrapper.metadata_registry == {}

    def test_store_kv_cache_success(self, dpu_config, mock_dpu_agent):
        """Test successful KV cache storage."""
        mock_agent_class, mock_instance = mock_dpu_agent
        mock_instance.store_kv.return_value = True

        wrapper = DPUAgentWrapper(dpu_config)

        # Create test tensors
        k_tensor = torch.randn(4, 32, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        v_tensor = torch.randn(4, 32, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")

        result = wrapper.store_kv_cache("test_key", k_tensor, v_tensor)

        assert result is True
        assert "test_key" in wrapper.metadata_registry

        # Verify metadata is stored correctly
        metadata = wrapper.metadata_registry["test_key"]
        assert metadata["k_shape"] == k_tensor.shape
        assert metadata["v_shape"] == v_tensor.shape
        assert metadata["dtype"] == k_tensor.dtype

    def test_store_kv_cache_failure(self, dpu_config, mock_dpu_agent):
        """Test KV cache storage failure."""
        mock_agent_class, mock_instance = mock_dpu_agent
        mock_instance.store_kv.side_effect = Exception("DMA operation failed")

        wrapper = DPUAgentWrapper(dpu_config)

        k_tensor = torch.randn(4, 32, 128, dtype=torch.float16)
        v_tensor = torch.randn(4, 32, 128, dtype=torch.float16)

        with pytest.raises(DPUOperationError, match="Failed to store KV cache"):
            wrapper.store_kv_cache("test_key", k_tensor, v_tensor)

    def test_retrieve_kv_cache_success(self, dpu_config, mock_dpu_agent):
        """Test successful KV cache retrieval."""
        mock_agent_class, mock_instance = mock_dpu_agent

        # Mock successful retrieval
        mock_k_data = torch.randn(4, 32, 128, dtype=torch.float16)
        mock_v_data = torch.randn(4, 32, 128, dtype=torch.float16)
        mock_instance.retrieve_kv.return_value = (mock_k_data, mock_v_data)

        wrapper = DPUAgentWrapper(dpu_config)

        # Pre-populate metadata
        wrapper.metadata_registry["test_key"] = {
            "k_shape": torch.Size([4, 32, 128]),
            "v_shape": torch.Size([4, 32, 128]),
            "dtype": torch.float16
        }

        result = wrapper.retrieve_kv_cache("test_key")

        assert result is not None
        k_tensor, v_tensor = result
        assert k_tensor.shape == torch.Size([4, 32, 128])
        assert v_tensor.shape == torch.Size([4, 32, 128])
        assert k_tensor.dtype == torch.float16

    def test_retrieve_kv_cache_not_found(self, dpu_config, mock_dpu_agent):
        """Test KV cache retrieval when key not found."""
        wrapper = DPUAgentWrapper(dpu_config)

        result = wrapper.retrieve_kv_cache("nonexistent_key")
        assert result is None

    def test_contains_key(self, dpu_config, mock_dpu_agent):
        """Test key existence check."""
        wrapper = DPUAgentWrapper(dpu_config)

        # Key not in registry
        assert wrapper.contains_key("test_key") is False

        # Add key to registry
        wrapper.metadata_registry["test_key"] = {"some": "metadata"}
        assert wrapper.contains_key("test_key") is True

    def test_remove_key_success(self, dpu_config, mock_dpu_agent):
        """Test successful key removal."""
        mock_agent_class, mock_instance = mock_dpu_agent
        mock_instance.remove_kv.return_value = True

        wrapper = DPUAgentWrapper(dpu_config)
        wrapper.metadata_registry["test_key"] = {"some": "metadata"}

        result = wrapper.remove_key("test_key")

        assert result is True
        assert "test_key" not in wrapper.metadata_registry

    def test_remove_key_not_found(self, dpu_config, mock_dpu_agent):
        """Test key removal when key not found."""
        wrapper = DPUAgentWrapper(dpu_config)

        result = wrapper.remove_key("nonexistent_key")
        assert result is False