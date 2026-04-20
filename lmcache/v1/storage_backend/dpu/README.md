# DPU Storage Backend for LMCache

A high-performance storage backend implementation for LMCache that leverages DPU (Data Processing Unit) hardware for efficient KV cache management with automatic fallback support.

## Features

- **High-Performance DPU Integration**: Direct integration with DPU hardware for maximum throughput
- **Automatic Fallback**: Seamlessly falls back to local storage when DPU is unavailable
- **Comprehensive Error Handling**: Robust error handling with configurable retry mechanisms
- **Memory Format Support**: Supports KV_2LTD memory format with efficient tensor operations
- **Configurable Parameters**: Extensive configuration options for fine-tuning performance

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/lmc_dpu_storage_backend.git
cd lmc_dpu_storage_backend

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e .[all]
```

### From PyPI

```bash
pip install lmc_dpu_storage_backend
```

## Quick Start

### 1. Configuration

Create a configuration file `dpu_config.yaml`:

```yaml
chunk_size: 256
local_device: "cuda"
pipelined_backend: false

storage_backend:
  type: "DPUStorageBackend"
  config:
    dpu_device_pci: "03:00.0"          # DPU PCI address
    max_concurrent_ops: 16             # Maximum concurrent operations
    connection_timeout: 5.0            # Connection timeout (seconds)
    retry_attempts: 3                  # Number of retry attempts
    fallback_enabled: true             # Enable fallback to local storage
    metadata_cache_size: 1000          # Metadata cache size

cache_manager:
  type: "LMCacheLocalCacheManagerV1"
  config:
    host_ip: "0.0.0.0"
    host_port: 7000
    local_cpu:
      type: "LocalCPUBackend"
      config:
        capacity_bytes: 2147483648  # 2GB
        device: "cpu"
```

### 2. Usage with vLLM

```python
import os
import sys

# Add the plugin to Python path if not installed
sys.path.insert(0, "/path/to/dpu/backend")

# Set configuration
os.environ['LMCACHE_CONFIG_FILE'] = '/path/to/dpu_config.yaml'

# Start vLLM with DPU backend
from vllm.entrypoints.cli.main import main as vllm_main
import json

sys.argv = [
    'vllm', 'serve', 'your-model-path',
    '--port', '8000',
    '--kv-transfer-config', json.dumps({
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both"
    })
]

vllm_main()
```

### 3. Direct API Usage

```python
from lmcache.v1.storage_backend.dpu_storage_backend import DPUStorageBackend
from lmcache.v1.storage_backend.dpu.config import DPUConfig

# Create DPU configuration
config = DPUConfig(
    dpu_device_pci="03:00.0",
    max_concurrent_ops=16,
    connection_timeout=5.0,
    retry_attempts=3,
    fallback_enabled=True,
    metadata_cache_size=1000
)

# Initialize backend (requires LMCache framework)
# backend = DPUStorageBackend(...)
```

## Configuration Options

### DPU Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpu_device_pci` | str | "03:00.0" | PCI address of the DPU device |
| `max_concurrent_ops` | int | 16 | Maximum concurrent operations (1-1000) |
| `connection_timeout` | float | 5.0 | Connection timeout in seconds (0.1-300.0) |
| `retry_attempts` | int | 3 | Number of retry attempts (0-100) |
| `fallback_enabled` | bool | true | Enable fallback to local storage |
| `metadata_cache_size` | int | 1000 | Metadata cache size (1-1000000) |

### PCI Address Formats

Supports both short and long PCI address formats:
- Short: `03:00.0`, `1a:02.1`
- Long: `0000:03:00.0`, `0001:1a:02.1`

## Architecture

### Components

1. **DPUStorageBackend**: Main backend implementation
2. **DPUAgentWrapper**: Wrapper for DPU API integration
3. **DPUConfig**: Configuration management with validation
4. **Exception Classes**: Comprehensive error handling

### Memory Management

- Supports KV_2LTD memory format
- Efficient tensor splitting and reconstruction
- Automatic memory allocation through LMCache framework
- Fallback storage for reliability

### Error Handling

- Graceful degradation when DPU is unavailable
- Configurable retry mechanisms
- Comprehensive logging for debugging
- Automatic fallback switching

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lmcache.v1.storage_backend.dpu --cov-report=html
```

### Code Quality

```bash
# Format code
black lmcache/
isort lmcache/

# Type checking
mypy lmcache/v1/storage_backend/dpu/
```

## Hardware Requirements

- **DPU Hardware**: Compatible DPU device with DOCA support
- **CUDA**: NVIDIA GPU with CUDA support
- **Memory**: Sufficient system memory for fallback storage
- **Network**: High-speed interconnect for DPU communication

## Performance

- **Latency**: Sub-millisecond KV cache operations
- **Throughput**: Optimized for high-concurrent workloads
- **Scalability**: Linear scaling with DPU resources
- **Fallback**: Minimal performance impact during fallback

## Troubleshooting

### Common Issues

1. **DPU Not Available**
   - Check PCI address configuration
   - Verify DPU drivers are installed
   - Ensure fallback is enabled

2. **Import Errors**
   - Install LMCache framework first
   - Check Python path configuration
   - Verify all dependencies are installed

3. **Performance Issues**
   - Tune `max_concurrent_ops` parameter
   - Check DPU resource utilization
   - Monitor network latency

### Debug Logging

```python
import logging
logging.getLogger('lmcache.v1.storage_backend.dpu').setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/lmc_dpu_storage_backend/issues)
- **Documentation**: [docs.lmcache.ai/dpu-backend](https://docs.lmcache.ai/dpu-backend)
- **Community**: [LMCache Discord](https://discord.gg/lmcache)