# tests/v1/storage_backend/dpu/test_config.py
import pytest
from lmcache.v1.storage_backend.dpu.config import DPUConfig, validate_dpu_config
from lmcache.v1.storage_backend.dpu.exceptions import DPUConfigurationError

def test_dpu_config_creation():
    """Test basic DPU configuration creation."""
    config_dict = {
        "dpu_device_pci": "03:00.0",
        "max_concurrent_ops": 16,
        "connection_timeout": 5.0,
        "retry_attempts": 3,
        "fallback_enabled": True,
        "metadata_cache_size": 1000,
    }

    dpu_config = DPUConfig.from_dict(config_dict)
    assert dpu_config.dpu_device_pci == "03:00.0"
    assert dpu_config.max_concurrent_ops == 16
    assert dpu_config.connection_timeout == 5.0
    assert dpu_config.retry_attempts == 3
    assert dpu_config.fallback_enabled is True
    assert dpu_config.metadata_cache_size == 1000

def test_dpu_config_validation():
    """Test DPU configuration validation."""
    # Valid config should pass
    valid_config = {
        "dpu_device_pci": "03:00.0",
        "max_concurrent_ops": 16,
    }
    validate_dpu_config(valid_config)

    # Invalid PCI address should fail
    invalid_config = {
        "dpu_device_pci": "invalid",
        "max_concurrent_ops": 16,
    }
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Invalid concurrent ops should fail
    invalid_ops_config = {
        "dpu_device_pci": "03:00.0",
        "max_concurrent_ops": 0,
    }
    with pytest.raises(DPUConfigurationError, match="max_concurrent_ops must be positive"):
        validate_dpu_config(invalid_ops_config)

def test_dpu_config_defaults():
    """Test default configuration values."""
    minimal_config = {"dpu_device_pci": "03:00.0"}
    dpu_config = DPUConfig.from_dict(minimal_config)

    # Check defaults are applied
    assert dpu_config.max_concurrent_ops == 16  # default
    assert dpu_config.connection_timeout == 5.0  # default
    assert dpu_config.retry_attempts == 3  # default
    assert dpu_config.fallback_enabled is True  # default
    assert dpu_config.metadata_cache_size == 1000  # default

def test_metadata_cache_size_validation():
    """Test metadata_cache_size validation."""
    base_config = {"dpu_device_pci": "03:00.0"}

    # Test negative values
    invalid_config = {**base_config, "metadata_cache_size": -1}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must be positive integer"):
        validate_dpu_config(invalid_config)

    # Test zero values
    invalid_config = {**base_config, "metadata_cache_size": 0}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must be positive integer"):
        validate_dpu_config(invalid_config)

    # Test wrong type (string)
    invalid_config = {**base_config, "metadata_cache_size": "1000"}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must be positive integer"):
        validate_dpu_config(invalid_config)

    # Test wrong type (float)
    invalid_config = {**base_config, "metadata_cache_size": 1000.5}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must be positive integer"):
        validate_dpu_config(invalid_config)

    # Test upper bound exceeded
    invalid_config = {**base_config, "metadata_cache_size": 1000001}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must not exceed 1000000"):
        validate_dpu_config(invalid_config)

    # Test valid values
    valid_config = {**base_config, "metadata_cache_size": 1}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "metadata_cache_size": 1000000}
    validate_dpu_config(valid_config)  # Should not raise

def test_fallback_enabled_validation():
    """Test fallback_enabled type validation."""
    base_config = {"dpu_device_pci": "03:00.0"}

    # Test invalid types
    invalid_config = {**base_config, "fallback_enabled": "true"}
    with pytest.raises(DPUConfigurationError, match="fallback_enabled must be a boolean value"):
        validate_dpu_config(invalid_config)

    invalid_config = {**base_config, "fallback_enabled": 1}
    with pytest.raises(DPUConfigurationError, match="fallback_enabled must be a boolean value"):
        validate_dpu_config(invalid_config)

    invalid_config = {**base_config, "fallback_enabled": 0}
    with pytest.raises(DPUConfigurationError, match="fallback_enabled must be a boolean value"):
        validate_dpu_config(invalid_config)

    invalid_config = {**base_config, "fallback_enabled": None}
    with pytest.raises(DPUConfigurationError, match="fallback_enabled must be a boolean value"):
        validate_dpu_config(invalid_config)

    # Test valid values
    valid_config = {**base_config, "fallback_enabled": True}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "fallback_enabled": False}
    validate_dpu_config(valid_config)  # Should not raise

def test_pci_address_edge_cases():
    """Test PCI address validation edge cases."""
    base_config = {"max_concurrent_ops": 16}

    # Test empty string
    invalid_config = {**base_config, "dpu_device_pci": ""}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address: "):
        validate_dpu_config(invalid_config)

    # Test special characters
    invalid_config = {**base_config, "dpu_device_pci": "03:00.0@"}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Test invalid format - missing dot
    invalid_config = {**base_config, "dpu_device_pci": "03:000"}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Test invalid format - too short
    invalid_config = {**base_config, "dpu_device_pci": "3:0.0"}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Test invalid format - too long function number
    invalid_config = {**base_config, "dpu_device_pci": "03:00.00"}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Test invalid hex characters
    invalid_config = {**base_config, "dpu_device_pci": "03:00.G"}
    with pytest.raises(DPUConfigurationError, match="Invalid PCI address"):
        validate_dpu_config(invalid_config)

    # Test boundary cases - valid formats
    valid_config = {**base_config, "dpu_device_pci": "00:00.0"}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "dpu_device_pci": "ff:ff.f"}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "dpu_device_pci": "0000:00:00.0"}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "dpu_device_pci": "ffff:ff:ff.f"}
    validate_dpu_config(valid_config)  # Should not raise

    # Test mixed case hex
    valid_config = {**base_config, "dpu_device_pci": "aB:Cd.E"}
    validate_dpu_config(valid_config)  # Should not raise

def test_error_message_validation():
    """Test specific error message content."""
    base_config = {"dpu_device_pci": "03:00.0"}

    # Test max_concurrent_ops error messages
    invalid_config = {**base_config, "max_concurrent_ops": 0}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "max_concurrent_ops must be positive integer" in str(exc_info.value)

    invalid_config = {**base_config, "max_concurrent_ops": "16"}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "max_concurrent_ops must be positive integer" in str(exc_info.value)

    # Test connection_timeout error messages
    invalid_config = {**base_config, "connection_timeout": 0}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "connection_timeout must be positive number" in str(exc_info.value)

    invalid_config = {**base_config, "connection_timeout": "5.0"}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "connection_timeout must be positive number" in str(exc_info.value)

    # Test retry_attempts error messages
    invalid_config = {**base_config, "retry_attempts": -1}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "retry_attempts must be non-negative integer" in str(exc_info.value)

    invalid_config = {**base_config, "retry_attempts": "3"}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "retry_attempts must be non-negative integer" in str(exc_info.value)

    # Test PCI address specific error message
    invalid_config = {"dpu_device_pci": "invalid_pci"}
    with pytest.raises(DPUConfigurationError) as exc_info:
        validate_dpu_config(invalid_config)
    assert "Invalid PCI address: invalid_pci" in str(exc_info.value)

def test_upper_bounds_validation():
    """Test upper bounds validation for numeric config values."""
    base_config = {"dpu_device_pci": "03:00.0"}

    # Test max_concurrent_ops upper bound
    invalid_config = {**base_config, "max_concurrent_ops": 1001}
    with pytest.raises(DPUConfigurationError, match="max_concurrent_ops must not exceed 1000"):
        validate_dpu_config(invalid_config)

    # Test connection_timeout upper bound
    invalid_config = {**base_config, "connection_timeout": 301.0}
    with pytest.raises(DPUConfigurationError, match="connection_timeout must not exceed 300 seconds"):
        validate_dpu_config(invalid_config)

    # Test retry_attempts upper bound
    invalid_config = {**base_config, "retry_attempts": 101}
    with pytest.raises(DPUConfigurationError, match="retry_attempts must not exceed 100"):
        validate_dpu_config(invalid_config)

    # Test metadata_cache_size upper bound
    invalid_config = {**base_config, "metadata_cache_size": 1000001}
    with pytest.raises(DPUConfigurationError, match="metadata_cache_size must not exceed 1000000"):
        validate_dpu_config(invalid_config)

    # Test boundary values (should pass)
    valid_config = {**base_config, "max_concurrent_ops": 1000}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "connection_timeout": 300.0}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "retry_attempts": 100}
    validate_dpu_config(valid_config)  # Should not raise

    valid_config = {**base_config, "metadata_cache_size": 1000000}
    validate_dpu_config(valid_config)  # Should not raise