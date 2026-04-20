# SPDX-License-Identifier: Apache-2.0

import asyncio
import pytest
import torch
from unittest.mock import Mock, patch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends


class TestDPUIntegration:
    """Test DPU backend integration with LMCache system."""

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_backend_creation_via_plugin_launcher(self, mock_wrapper_class):
        """Test DPU backend creation through storage plugin launcher."""
        # Create config with DPU plugin configuration
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024  # 1MB
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        # Configure DPU plugin
        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        # Create metadata
        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        # Mock torch.cuda.is_available to return True for testing
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # Verify DPU backend was created
            assert "dpu" in storage_backends
            dpu_backend = storage_backends["dpu"]

            # Verify it's the correct type
            from lmcache.v1.storage_backend.dpu_storage_backend import DPUStorageBackend
            assert isinstance(dpu_backend, DPUStorageBackend)

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_backend_with_local_cpu_dependency(self, mock_wrapper_class):
        """Test DPU backend works with LocalCPUBackend dependency."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # Verify both LocalCPUBackend and DPUStorageBackend exist
            assert "LocalCPUBackend" in storage_backends
            assert "dpu" in storage_backends

            # Verify DPU backend has reference to LocalCPU backend
            dpu_backend = storage_backends["dpu"]
            local_cpu_backend = storage_backends["LocalCPUBackend"]

            assert dpu_backend.local_cpu_backend == local_cpu_backend

    def test_dpu_plugin_missing_configuration(self):
        """Test error handling when DPU plugin configuration is missing."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        # Missing plugin configuration
        config.storage_plugins = ["dpu"]
        config.extra_config = {}  # Missing DPU configuration

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should not be created due to missing config
            assert "dpu" not in storage_backends
            # But LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_plugin_with_audit_backend(self, mock_wrapper_class):
        """Test DPU backend integration with audit wrapper."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
            "audit_backend_enabled": True,  # Enable audit wrapper
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # Verify DPU backend exists and is wrapped with audit
            assert "dpu" in storage_backends
            dpu_backend = storage_backends["dpu"]

            # Should be wrapped with AuditBackend (not direct DPUStorageBackend)
            from lmcache.v1.storage_backend.audit_backend import AuditBackend
            assert isinstance(dpu_backend, AuditBackend)

    def test_dpu_plugin_missing_module_path(self):
        """Test error handling when DPU plugin module_path is missing."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            # Missing module_path
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should not be created due to missing module_path
            assert "dpu" not in storage_backends
            # But LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    def test_dpu_plugin_missing_class_name(self):
        """Test error handling when DPU plugin class_name is missing."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            # Missing class_name
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should not be created due to missing class_name
            assert "dpu" not in storage_backends
            # But LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_plugin_with_scheduler_role(self, mock_wrapper_class):
        """Test DPU plugin behavior with scheduler role (no local_cpu_backend)."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "scheduler"  # Scheduler role

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # For scheduler role, DPU backend creation should be attempted
            # but with local_cpu_backend=None
            if "dpu" in storage_backends:
                dpu_backend = storage_backends["dpu"]
                assert dpu_backend.local_cpu_backend is None

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_multiple_storage_plugins_including_dpu(self, mock_wrapper_class):
        """Test DPU plugin creation alongside other plugins."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        # Multiple plugins including DPU
        config.storage_plugins = ["dpu", "test_plugin"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
            # Missing test_plugin config - should be skipped
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should be created successfully
            assert "dpu" in storage_backends
            # test_plugin should not be created due to missing config
            assert "test_plugin" not in storage_backends
            # LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_plugin_import_error_handling(self, mock_wrapper_class):
        """Test error handling when DPU plugin module import fails."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "nonexistent.module.path",  # Invalid module
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should not be created due to import error
            assert "dpu" not in storage_backends
            # But LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_plugin_class_not_found_error(self, mock_wrapper_class):
        """Test error handling when DPU plugin class is not found in module."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 1024 * 1024
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "NonExistentClass",  # Invalid class name
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should not be created due to class not found error
            assert "dpu" not in storage_backends
            # But LocalCPU backend should still exist
            assert "LocalCPUBackend" in storage_backends

    @patch('lmcache.v1.storage_backend.dpu_storage_backend.DPUAgentWrapper')
    def test_dpu_plugin_without_local_cpu_disabled(self, mock_wrapper_class):
        """Test DPU plugin when local_cpu is disabled (max_local_cpu_size=0)."""
        config = Mock(spec=LMCacheEngineConfig)
        config.enable_pd = False
        config.local_cpu = True
        config.max_local_cpu_size = 0  # Disabled LocalCPU backend
        config.enable_p2p = False
        config.local_disk = False
        config.gds_path = None
        config.maru_path = None
        config.remote_storage_plugins = None
        config.remote_url = None
        config.max_local_disk_size = 0

        config.storage_plugins = ["dpu"]
        config.extra_config = {
            "storage_plugin.dpu.module_path": "lmcache.v1.storage_backend.dpu_storage_backend",
            "storage_plugin.dpu.class_name": "DPUStorageBackend",
        }

        metadata = Mock(spec=LMCacheMetadata)
        metadata.role = "worker"

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0):

            storage_backends = CreateStorageBackends(
                config=config,
                metadata=metadata,
                loop=asyncio.get_event_loop(),
                dst_device="cuda",
            )

            # DPU backend should be created with local_cpu_backend=None
            if "dpu" in storage_backends:
                dpu_backend = storage_backends["dpu"]
                assert dpu_backend.local_cpu_backend is None

            # LocalCPU backend should not exist
            assert "LocalCPUBackend" not in storage_backends