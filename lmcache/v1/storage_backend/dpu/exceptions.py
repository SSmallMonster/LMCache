# SPDX-License-Identifier: Apache-2.0

"""DPU-specific exception classes."""


class DPUError(Exception):
    """Base exception for DPU-related errors."""
    pass


class DPUConfigurationError(DPUError):
    """Exception raised for DPU configuration errors."""
    pass


class DPUConnectionError(DPUError):
    """Exception raised for DPU connection errors."""
    pass


class DPUOperationError(DPUError):
    """Exception raised for DPU operation failures."""
    pass


class DPUTimeoutError(DPUError):
    """Exception raised for DPU operation timeouts."""
    pass