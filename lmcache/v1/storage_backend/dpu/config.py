# SPDX-License-Identifier: Apache-2.0

"""DPU configuration management."""

import re
from dataclasses import dataclass
from typing import Dict, Any

from .exceptions import DPUConfigurationError


@dataclass
class DPUConfig:
    """Configuration for DPU storage backend.

    This class holds all configuration parameters needed to connect to and
    communicate with a DPU (Data Processing Unit) storage backend.

    Attributes:
        dpu_device_pci (str): PCI address of the DPU device (e.g., "03:00.0" or "0000:03:00.0").
                             Must be a valid PCI bus address format.
        max_concurrent_ops (int): Maximum number of concurrent operations allowed.
                                 Must be positive integer, maximum 1000. Default: 16.
        connection_timeout (float): Timeout in seconds for establishing connections.
                                   Must be positive number, maximum 300 seconds. Default: 5.0.
        retry_attempts (int): Number of retry attempts for failed operations.
                             Must be non-negative integer, maximum 100. Default: 3.
        fallback_enabled (bool): Whether to enable fallback mechanisms when DPU is unavailable.
                                Default: True.
        metadata_cache_size (int): Size of metadata cache in entries.
                                  Must be positive integer, maximum 1,000,000. Default: 1000.
    """

    # Required fields
    dpu_device_pci: str

    # Optional fields with defaults
    max_concurrent_ops: int = 16
    connection_timeout: float = 5.0
    retry_attempts: int = 3
    fallback_enabled: bool = True
    metadata_cache_size: int = 1000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DPUConfig":
        """Create DPUConfig from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
                        Must contain 'dpu_device_pci' key. Other fields are optional.

        Returns:
            DPUConfig: Validated configuration object.

        Raises:
            DPUConfigurationError: If required fields are missing or validation fails.
        """
        # Validate required fields
        if "dpu_device_pci" not in config_dict:
            raise DPUConfigurationError("Missing required field: dpu_device_pci")

        # Create config with defaults
        config = cls(
            dpu_device_pci=config_dict["dpu_device_pci"],
            max_concurrent_ops=config_dict.get("max_concurrent_ops", 16),
            connection_timeout=config_dict.get("connection_timeout", 5.0),
            retry_attempts=config_dict.get("retry_attempts", 3),
            fallback_enabled=config_dict.get("fallback_enabled", True),
            metadata_cache_size=config_dict.get("metadata_cache_size", 1000),
        )

        # Validate the config
        validate_dpu_config_object(config)
        return config


def validate_dpu_config(config_dict: Dict[str, Any]) -> None:
    """Validate DPU configuration dictionary.

    Args:
        config_dict: Dictionary containing DPU configuration parameters.
                    Required keys: dpu_device_pci
                    Optional keys: max_concurrent_ops, connection_timeout,
                                 retry_attempts, fallback_enabled, metadata_cache_size

    Raises:
        DPUConfigurationError: If any configuration parameter is invalid.
    """
    # Validate PCI address format
    pci_address = config_dict.get("dpu_device_pci", "")
    if not _is_valid_pci_address(pci_address):
        raise DPUConfigurationError(f"Invalid PCI address: {pci_address}")

    # Validate max_concurrent_ops
    max_ops = config_dict.get("max_concurrent_ops", 16)
    if not isinstance(max_ops, int) or max_ops <= 0:
        raise DPUConfigurationError("max_concurrent_ops must be positive integer")
    if max_ops > 1000:  # Upper bound check
        raise DPUConfigurationError("max_concurrent_ops must not exceed 1000")

    # Validate connection_timeout
    timeout = config_dict.get("connection_timeout", 5.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise DPUConfigurationError("connection_timeout must be positive number")
    if timeout > 300.0:  # Upper bound check (5 minutes max)
        raise DPUConfigurationError("connection_timeout must not exceed 300 seconds")

    # Validate retry_attempts
    retries = config_dict.get("retry_attempts", 3)
    if not isinstance(retries, int) or retries < 0:
        raise DPUConfigurationError("retry_attempts must be non-negative integer")
    if retries > 100:  # Upper bound check
        raise DPUConfigurationError("retry_attempts must not exceed 100")

    # Validate fallback_enabled
    fallback = config_dict.get("fallback_enabled", True)
    if not isinstance(fallback, bool):
        raise DPUConfigurationError("fallback_enabled must be a boolean value")

    # Validate metadata_cache_size
    cache_size = config_dict.get("metadata_cache_size", 1000)
    if not isinstance(cache_size, int) or cache_size <= 0:
        raise DPUConfigurationError("metadata_cache_size must be positive integer")
    if cache_size > 1000000:  # Upper bound check (1 million entries max)
        raise DPUConfigurationError("metadata_cache_size must not exceed 1000000")


def validate_dpu_config_object(config: DPUConfig) -> None:
    """Validate DPUConfig object.

    Args:
        config: DPUConfig object to validate.

    Raises:
        DPUConfigurationError: If any configuration parameter is invalid.
    """
    if not _is_valid_pci_address(config.dpu_device_pci):
        raise DPUConfigurationError(f"Invalid PCI address: {config.dpu_device_pci}")

    if config.max_concurrent_ops <= 0:
        raise DPUConfigurationError("max_concurrent_ops must be positive")
    if config.max_concurrent_ops > 1000:
        raise DPUConfigurationError("max_concurrent_ops must not exceed 1000")

    if config.connection_timeout <= 0:
        raise DPUConfigurationError("connection_timeout must be positive")
    if config.connection_timeout > 300.0:
        raise DPUConfigurationError("connection_timeout must not exceed 300 seconds")

    if config.retry_attempts < 0:
        raise DPUConfigurationError("retry_attempts must be non-negative")
    if config.retry_attempts > 100:
        raise DPUConfigurationError("retry_attempts must not exceed 100")

    if not isinstance(config.fallback_enabled, bool):
        raise DPUConfigurationError("fallback_enabled must be a boolean value")

    if config.metadata_cache_size <= 0:
        raise DPUConfigurationError("metadata_cache_size must be positive")
    if config.metadata_cache_size > 1000000:
        raise DPUConfigurationError("metadata_cache_size must not exceed 1000000")


def _is_valid_pci_address(address: str) -> bool:
    """Validate PCI address format.

    Supports two standard PCI address formats:
    - Short format: XX:XX.X (e.g., "03:00.0", "1a:02.1")
    - Long format: XXXX:XX:XX.X (e.g., "0000:03:00.0", "0001:1a:02.1")

    Where:
    - X represents hexadecimal digits (0-9, a-f, A-F)
    - The short format omits the domain prefix (assumes 0000)
    - Function number is a single hex digit after the dot

    Args:
        address: PCI address string to validate.

    Returns:
        bool: True if the address matches valid PCI format, False otherwise.

    Examples:
        >>> _is_valid_pci_address("03:00.0")
        True
        >>> _is_valid_pci_address("0000:1a:02.1")
        True
        >>> _is_valid_pci_address("invalid")
        False
    """
    # Basic PCI address format: XX:XX.X or XXXX:XX:XX.X
    pattern = r'^([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]|[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])$'
    return bool(re.match(pattern, address))