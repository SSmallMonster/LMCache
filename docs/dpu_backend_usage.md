# DPU Storage Backend Usage Guide

## Overview

The DPU (Data Processing Unit) storage backend provides immediate KV cache offloading and on-demand retrieval for LMCache, leveraging DOCA DMA operations for high-performance data transfer.

## Features

- **Immediate Offloading**: KV caches are immediately transferred to DPU storage after prefill
- **On-Demand Retrieval**: KV caches are retrieved from DPU storage as needed during decode
- **Fallback Mechanism**: Automatic fallback to local storage if DPU becomes unavailable
- **Integration**: Full integration with LMCache's plugin system and memory management

## Configuration

### Basic Configuration

Create a configuration file (e.g., `dpu_config.yaml`):

```yaml
# Enable DPU storage backend
storage_plugins: ["dpu"]

# Plugin configuration
extra_config:
  storage_plugin.dpu.module_path: "lmcache.v1.storage_backend.dpu_storage_backend"
  storage_plugin.dpu.class_name: "DPUStorageBackend"

  # DPU-specific settings
  dpu_device_pci: "03:00.0"          # DPU PCIe address
  max_concurrent_ops: 16              # Maximum concurrent DMA operations
  connection_timeout: 5.0             # Connection timeout in seconds
  retry_attempts: 3                   # Number of retry attempts
  fallback_enabled: true              # Enable fallback to local storage
  metadata_cache_size: 1000           # Local metadata cache size

# Required: Enable local CPU backend as dependency
local_cpu: true
max_local_cpu_size: 1073741824        # 1GB local CPU memory
```

### Advanced Configuration

```yaml
storage_plugins: ["dpu"]

extra_config:
  storage_plugin.dpu.module_path: "lmcache.v1.storage_backend.dpu_storage_backend"
  storage_plugin.dpu.class_name: "DPUStorageBackend"

  # DPU settings
  dpu_device_pci: "03:00.0"
  max_concurrent_ops: 32              # Higher concurrency for better performance
  connection_timeout: 10.0            # Longer timeout for unstable connections
  retry_attempts: 5                   # More retries for reliability
  fallback_enabled: true
  metadata_cache_size: 5000           # Larger cache for better performance

  # Enable audit logging
  audit_backend_enabled: true

# Local backends
local_cpu: true
max_local_cpu_size: 2147483648        # 2GB local CPU memory

# Optional: Enable other backends
local_disk: true
max_local_disk_size: 10737418240      # 10GB local disk storage
```

## Usage

### With vLLM Integration

```python
import asyncio
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend import CreateStorageBackends

# Load configuration
config = LMCacheEngineConfig.from_file("dpu_config.yaml")

# Create storage backends
loop = asyncio.get_event_loop()
storage_backends = CreateStorageBackends(
    config=config,
    metadata=metadata,
    loop=loop,
    dst_device="cuda:0"
)

# DPU backend will be available as storage_backends["dpu"]
dpu_backend = storage_backends["dpu"]
```

### Direct Usage

```python
from lmcache.v1.storage_backend.dpu_storage_backend import DPUStorageBackend
from lmcache.v1.storage_backend.dpu.config import DPUConfig

# Create DPU configuration
dpu_config = DPUConfig.from_dict({
    "dpu_device_pci": "03:00.0",
    "max_concurrent_ops": 16,
    "fallback_enabled": True,
})

# Create DPU backend
dpu_backend = DPUStorageBackend(
    dst_device="cuda:0",
    config=config,
    metadata=metadata,
    local_cpu_backend=local_cpu_backend,
)

# Use backend for KV cache operations
# Store operation
futures = dpu_backend.batched_submit_put_task([key], [memory_obj])

# Retrieve operation
retrieved_obj = dpu_backend.get_blocking(key)

# Check existence
exists = dpu_backend.contains(key)
```

## Error Handling

The DPU backend includes comprehensive error handling:

1. **Connection Failures**: Automatic retry with configurable attempts
2. **DPU Unavailability**: Fallback to local storage when enabled
3. **Operation Failures**: Graceful degradation with logging
4. **Configuration Errors**: Clear error messages for invalid settings

## Performance Considerations

- **Concurrency**: Adjust `max_concurrent_ops` based on DPU capabilities
- **Memory**: Ensure sufficient local CPU memory for fallback operations
- **Network**: DPU operations depend on PCIe bandwidth and latency
- **Caching**: Larger `metadata_cache_size` improves lookup performance

## Troubleshooting

### Common Issues

1. **DPU Not Found**
   - Verify `dpu_device_pci` setting matches actual DPU PCIe address
   - Check DPU driver installation and device accessibility

2. **Connection Timeouts**
   - Increase `connection_timeout` value
   - Check DPU load and network conditions

3. **Fallback Mode Activated**
   - Check DPU availability and logs
   - Verify DOCA library installation and configuration

4. **Performance Issues**
   - Monitor DPU utilization and concurrent operations
   - Adjust `max_concurrent_ops` for optimal throughput
   - Consider memory allocation patterns and tensor sizes

### Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('lmcache.v1.storage_backend.dpu_storage_backend').setLevel(logging.DEBUG)
logging.getLogger('lmcache.v1.storage_backend.dpu.agent_wrapper').setLevel(logging.DEBUG)
```

## Dependencies

- **DOCA SDK**: Required for DPU communication
- **PyTorch**: For tensor operations
- **LMCache**: Core LMCache framework
- **Local CPU Backend**: Required dependency for memory allocation

## Limitations

- Synchronous operations only (no async put operations yet)
- Requires DOCA-compatible DPU hardware
- Fallback storage is memory-only (not persistent)
- Limited to single-DPU configurations in Phase 2