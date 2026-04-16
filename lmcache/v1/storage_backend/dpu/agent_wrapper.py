# lmcache/v1/storage_backend/dpu/agent_wrapper.py
# SPDX-License-Identifier: Apache-2.0

"""DPU Agent wrapper for C API integration."""

import logging
from typing import Dict, Optional, Tuple, Any
import torch

from .config import DPUConfig
from .exceptions import DPUConnectionError, DPUOperationError

logger = logging.getLogger(__name__)

try:
    # Import DPU C API - this will be the actual implementation
    from dpu_cache._api import dpu_agent
    from dpu_cache._api import dpu_agent_config
    DPU_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DPU API not available: {e}")
    # Mock implementations for development/testing
    dpu_agent = None
    dpu_agent_config = None
    DPU_API_AVAILABLE = False


class DPUAgentWrapper:
    """
    Wrapper for DPU C API providing Python interface.

    Encapsulates DOCA DMA operations and manages DPU device lifecycle.
    Based on nixl backend implementation pattern for direct C library calls.
    """

    def __init__(self, config: DPUConfig):
        """
        Initialize DPU agent wrapper.

        Args:
            config: DPU configuration object

        Raises:
            DPUConnectionError: If DPU initialization fails
        """
        self.config = config
        self.metadata_registry: Dict[str, Dict[str, Any]] = {}
        self.agent = None

        if not DPU_API_AVAILABLE:
            logger.warning("DPU API not available, using mock implementation")
            self.agent = MockDPUAgent(config)
        else:
            try:
                # Create DPU agent configuration
                agent_config = dpu_agent_config(
                    device_pci=config.dpu_device_pci,
                    max_concurrent_ops=config.max_concurrent_ops
                )

                # Initialize DPU agent
                self.agent = dpu_agent(agent_config)
                logger.info(f"Initialized DPU agent on device {config.dpu_device_pci}")

            except Exception as e:
                raise DPUConnectionError(f"Failed to initialize DPU agent: {e}") from e

    def store_kv_cache(
        self,
        key_id: str,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor
    ) -> bool:
        """
        Store KV cache tensors to DPU.

        Process:
        1. Get tensor GPU memory addresses
        2. Create DOCA memory map
        3. Submit DMA copy task
        4. Update metadata registry

        Args:
            key_id: Unique identifier for the KV cache
            k_tensor: Key tensor to store
            v_tensor: Value tensor to store

        Returns:
            True if storage successful, False otherwise

        Raises:
            DPUOperationError: If storage operation fails
        """
        try:
            # Validate tensors are on same device and have same dtype
            if k_tensor.device != v_tensor.device:
                raise ValueError("K and V tensors must be on same device")
            if k_tensor.dtype != v_tensor.dtype:
                raise ValueError("K and V tensors must have same dtype")

            # Store metadata first
            metadata = {
                "k_shape": k_tensor.shape,
                "v_shape": v_tensor.shape,
                "dtype": k_tensor.dtype,
                "device": str(k_tensor.device),
            }

            # Perform DMA operation
            result = self.agent.store_kv(key_id, k_tensor, v_tensor)

            if result:
                # Update local metadata registry on success
                self.metadata_registry[key_id] = metadata
                logger.debug(f"Successfully stored KV cache for key: {key_id}")
                return True
            else:
                logger.warning(f"DPU storage returned failure for key: {key_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to store KV cache for key {key_id}: {e}")
            raise DPUOperationError(f"Failed to store KV cache: {e}") from e

    def retrieve_kv_cache(
        self,
        key_id: str,
        target_device: str = "cuda"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV cache tensors from DPU.

        Process:
        1. Check metadata registry
        2. Allocate memory on target device
        3. Submit DMA copy task
        4. Return tensor objects

        Args:
            key_id: Unique identifier for the KV cache
            target_device: Device to retrieve tensors to

        Returns:
            Tuple of (k_tensor, v_tensor) if found, None otherwise

        Raises:
            DPUOperationError: If retrieval operation fails
        """
        # Check if key exists in metadata registry
        if key_id not in self.metadata_registry:
            logger.debug(f"Key not found in metadata registry: {key_id}")
            return None

        try:
            metadata = self.metadata_registry[key_id]

            # Retrieve from DPU
            result = self.agent.retrieve_kv(key_id, target_device)

            if result is None:
                logger.warning(f"DPU retrieval returned None for key: {key_id}")
                return None

            k_tensor, v_tensor = result

            # Validate retrieved tensors match metadata
            if (k_tensor.shape != metadata["k_shape"] or
                v_tensor.shape != metadata["v_shape"] or
                k_tensor.dtype != metadata["dtype"]):
                raise DPUOperationError(
                    f"Retrieved tensor metadata mismatch for key {key_id}"
                )

            logger.debug(f"Successfully retrieved KV cache for key: {key_id}")
            return (k_tensor, v_tensor)

        except Exception as e:
            logger.error(f"Failed to retrieve KV cache for key {key_id}: {e}")
            raise DPUOperationError(f"Failed to retrieve KV cache: {e}") from e

    def contains_key(self, key_id: str) -> bool:
        """
        Check if DPU contains the specified key.

        Args:
            key_id: Unique identifier to check

        Returns:
            True if key exists in DPU, False otherwise
        """
        return key_id in self.metadata_registry

    def remove_key(self, key_id: str) -> bool:
        """
        Remove KV cache from DPU.

        Args:
            key_id: Unique identifier to remove

        Returns:
            True if removal successful, False otherwise
        """
        if key_id not in self.metadata_registry:
            logger.debug(f"Key not found for removal: {key_id}")
            return False

        try:
            # Remove from DPU
            result = self.agent.remove_kv(key_id)

            if result:
                # Remove from local metadata registry
                del self.metadata_registry[key_id]
                logger.debug(f"Successfully removed KV cache for key: {key_id}")
                return True
            else:
                logger.warning(f"DPU removal returned failure for key: {key_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove KV cache for key {key_id}: {e}")
            return False


class MockDPUAgent:
    """Mock DPU agent for testing when DPU API is not available."""

    def __init__(self, config: DPUConfig):
        self.config = config
        self._storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        logger.info("Initialized Mock DPU Agent")

    def store_kv(self, key_id: str, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> bool:
        """Mock store operation."""
        # Clone tensors to simulate DPU storage
        self._storage[key_id] = (k_tensor.clone(), v_tensor.clone())
        return True

    def retrieve_kv(self, key_id: str, target_device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Mock retrieve operation."""
        if key_id not in self._storage:
            return None

        k_tensor, v_tensor = self._storage[key_id]
        # Move to target device if different
        if str(k_tensor.device) != target_device:
            k_tensor = k_tensor.to(target_device)
            v_tensor = v_tensor.to(target_device)

        return (k_tensor, v_tensor)

    def remove_kv(self, key_id: str) -> bool:
        """Mock remove operation."""
        if key_id in self._storage:
            del self._storage[key_id]
            return True
        return False