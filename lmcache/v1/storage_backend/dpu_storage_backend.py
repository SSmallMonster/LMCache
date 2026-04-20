# SPDX-License-Identifier: Apache-2.0
"""
DPU Storage Backend Implementation for LMCache.

This module implements the DPU storage backend that provides high-performance
KV cache storage using DPU (Data Processing Unit) hardware with fallback
to local CPU storage for reliability.

Key Features:
- Immediate KV cache offloading to DPU
- On-demand retrieval from DPU
- Fallback mechanism for DPU failures
- Tensor conversion between MemoryObj and KV tensors
- Comprehensive error handling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from dpu_cache import DPUConfig,DPUAgent

logger = logging.getLogger(__name__)


class DPUStorageBackend(StoragePluginInterface):
    """
    DPU storage backend that implements the LMCache StoragePluginInterface.

    This backend provides high-performance KV cache storage using DPU hardware
    with fallback to local CPU storage for reliability.

    Features:
    - Immediate offloading: KV caches are immediately stored to DPU
    - On-demand retrieval: KV caches are retrieved from DPU when needed
    - Fallback mechanism: Falls back to local storage on DPU failures
    - Error handling: Comprehensive error handling and recovery
    """

    def __init__(
        self,
        dst_device: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_cpu_backend: LocalCPUBackend,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Initialize the DPU storage backend.

        Args:
            dst_device: Target device for memory allocation (e.g., "cuda", "cpu")
            config: LMCache engine configuration
            metadata: LMCache metadata
            local_cpu_backend: Local CPU backend for fallback storage
            loop: Event loop for async operations (optional)
        """
        super().__init__()

        self.dst_device = dst_device
        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop or asyncio.get_event_loop()

        # DPU configuration from config
        dpu_config_dict = config.extra_config or {}
        self.dpu_config = DPUConfig(
            dpu_device_pci=dpu_config_dict.get("dpu_device_pci", "03:00.0"),
            dpu_ip=dpu_config_dict.get("dpu_ip", "127.0.0.1"),
            gpu_id=dpu_config_dict.get("gpu_id", 0),
            max_concurrent_ops=dpu_config_dict.get("max_concurrent_ops", 16)
        )

        # Initialize DPU agent wrapper
        try:
            self.dpu_agent = DPUAgent(self.dpu_config)
            self.dpu_available = True
            logger.info(f"DPU storage backend initialized with device {self.dpu_config.dpu_device_pci}")
        except Exception as e:
            logger.error(f"Failed to initialize DPU agent: {e}")
            if self.dpu_config.fallback_enabled:
                self.dpu_available = False
                logger.warning("DPU unavailable, will use fallback storage")
            else:
                raise

        # Fallback storage for when DPU is unavailable
        self.fallback_storage: Dict[str, MemoryObj] = {}

    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check if a key exists in the storage backend.

        Args:
            key: The cache engine key to check

        Returns:
            True if the key exists, False otherwise
        """
        key_str = str(key)

        # Check DPU first if available
        if self.dpu_available:
            try:
                return self.dpu_agent.contains_key(key_str)
            except Exception as e:
                logger.error(f"Error checking key in DPU: {e}")
                self._handle_dpu_error(e)

        # Check fallback storage
        return key_str in self.fallback_storage

    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        objs: List[MemoryObj]
    ) -> Optional[List]:
        """
        Submit a batched put task to store KV caches.

        This method implements immediate offloading - KV caches are immediately
        stored to the DPU storage backend.

        Args:
            keys: List of cache engine keys
            objs: List of memory objects to store

        Returns:
            None for synchronous operations, List of tasks for async operations
        """
        if len(keys) != len(objs):
            raise ValueError("Keys and objects lists must have the same length")

        for key, obj in zip(keys, objs):
            self._store_single_object(key, obj)

        # Return None for synchronous operations
        return None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Retrieve a memory object from storage (blocking operation).

        This method implements on-demand retrieval - KV caches are retrieved
        from DPU storage when needed.

        Args:
            key: The cache engine key to retrieve

        Returns:
            The memory object if found, None otherwise
        """
        key_str = str(key)

        # Try to retrieve from DPU first
        if self.dpu_available:
            try:
                result = self.dpu_agent.retrieve_kv_cache(key_str, self.dst_device)
                if result is not None:
                    k_tensor, v_tensor = result
                    return self._create_memory_obj_from_kv(k_tensor, v_tensor)
            except Exception as e:
                logger.error(f"Error retrieving from DPU: {e}")
                self._handle_dpu_error(e)

        # Try fallback storage
        return self._fallback_get(key_str)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check if a key exists in pending put tasks.

        For DPU backend, operations are synchronous, so this always returns False.

        Args:
            key: The cache engine key to check

        Returns:
            False (no async put tasks for DPU backend)
        """
        return False

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Pin a memory object in storage.

        For DPU backend, this is a no-op since DPU manages its own memory.

        Args:
            key: The cache engine key to pin

        Returns:
            True (always succeeds)
        """
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Unpin a memory object in storage.

        For DPU backend, this is a no-op since DPU manages its own memory.

        Args:
            key: The cache engine key to unpin

        Returns:
            True (always succeeds)
        """
        return True

    def remove(self, key: CacheEngineKey) -> bool:
        """
        Remove a memory object from storage.

        Args:
            key: The cache engine key to remove

        Returns:
            True if the key was removed or didn't exist, False on error
        """
        key_str = str(key)

        # Try to remove from DPU first
        if self.dpu_available:
            try:
                result = self.dpu_agent.remove_key(key_str)
                if result:
                    return True
            except Exception as e:
                logger.error(f"Error removing key from DPU: {e}")
                self._handle_dpu_error(e)

        # Remove from fallback storage
        if key_str in self.fallback_storage:
            del self.fallback_storage[key_str]
            return True

        return False

    def get_allocator_backend(self):
        """
        Get the allocator backend for memory allocation.

        Returns:
            The local CPU backend which provides memory allocation
        """
        return self.local_cpu_backend

    def close(self) -> None:
        """
        Close the storage backend and cleanup resources.

        This is currently a no-op for DPU backend as the DPU agent
        handles resource cleanup automatically.
        """
        # For now, this is a no-op since DPU agent handles cleanup
        # In future versions, we might need to explicitly close DPU connections
        pass

    def touch_cache(self, key: CacheEngineKey) -> None:
        """
        Touch a cache entry to update its access time for cache policies.

        For DPU backend, this is a no-op since DPU manages its own cache policy.

        Args:
            key: The cache engine key to touch
        """
        # No-op for DPU backend - DPU manages its own cache policy
        pass

    # Helper methods

    def _store_single_object(self, key: CacheEngineKey, obj: MemoryObj) -> None:
        """
        Store a single memory object.

        Args:
            key: The cache engine key
            obj: The memory object to store
        """
        key_str = str(key)

        # Try to store in DPU first
        if self.dpu_available:
            try:
                k_tensor, v_tensor = self._extract_kv_tensors(obj)
                success = self.dpu_agent.store_kv_cache(key_str, k_tensor, v_tensor)
                if success:
                    return
            except Exception as e:
                logger.error(f"Error storing to DPU: {e}")
                self._handle_dpu_error(e)

        # Fallback to local storage
        self._fallback_store(key_str, obj)

    def _extract_kv_tensors(self, obj: MemoryObj) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K and V tensors from a memory object.

        Assumes the memory object uses KV_2LTD format where the tensor contains
        flattened K and V tensors concatenated together.

        Args:
            obj: The memory object containing KV cache data

        Returns:
            Tuple of (k_tensor, v_tensor)
        """
        if obj.meta.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(f"Unsupported memory format: {obj.meta.fmt}")

        # For KV_2LTD format, we need to use raw_tensor (the flattened data)
        # instead of tensor (which is already reshaped)
        raw_tensor = obj.raw_tensor
        if raw_tensor is None:
            raise ValueError("Memory object has no raw tensor data")

        # For KV_2LTD format, the raw tensor contains flattened K and V concatenated
        # We need to split it back into K and V tensors
        total_size = raw_tensor.numel()
        if total_size % 2 != 0:
            raise ValueError("KV tensor size must be even for K/V split")

        k_size = total_size // 2
        k_tensor_flat = raw_tensor[:k_size]
        v_tensor_flat = raw_tensor[k_size:total_size]

        # Reshape to the original logical shape
        k_tensor = k_tensor_flat.view(obj.meta.shape)
        v_tensor = v_tensor_flat.view(obj.meta.shape)

        return k_tensor, v_tensor

    def _create_memory_obj_from_kv(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor
    ) -> MemoryObj:
        """
        Create a memory object from K and V tensors.

        Args:
            k_tensor: The key tensor
            v_tensor: The value tensor

        Returns:
            A memory object in KV_2LTD format
        """
        # Get allocator from local CPU backend
        allocator = self.local_cpu_backend.get_memory_allocator()

        # Create memory object with the same shape as K tensor
        mem_obj = allocator.allocate(
            k_tensor.shape,
            k_tensor.dtype,
            fmt=MemoryFormat.KV_2LTD
        )

        # Concatenate K and V tensors and copy to memory object
        kv_concatenated = torch.cat([k_tensor.flatten(), v_tensor.flatten()])
        mem_obj.tensor.copy_(kv_concatenated)

        return mem_obj

    def _fallback_store(self, key_str: str, obj: MemoryObj) -> None:
        """
        Store object in fallback local storage.

        Args:
            key_str: The string key
            obj: The memory object to store
        """
        # Store in local fallback storage
        self.fallback_storage[key_str] = obj
        logger.debug(f"Stored key {key_str} in fallback storage")

    def _fallback_get(self, key_str: str) -> Optional[MemoryObj]:
        """
        Get object from fallback local storage.

        Args:
            key_str: The string key

        Returns:
            The memory object if found, None otherwise
        """
        return self.fallback_storage.get(key_str)

    def _handle_dpu_error(self, error: Exception) -> None:
        """
        Handle DPU errors and potentially disable DPU.

        Args:
            error: The error that occurred
        """
        if self.dpu_config.fallback_enabled and self.dpu_available:
            logger.warning(f"DPU error occurred: {error}. Disabling DPU and using fallback.")
            self.dpu_available = False
        else:
            logger.error(f"DPU error occurred and fallback is disabled: {error}")
            # Don't re-raise the error to allow graceful degradation