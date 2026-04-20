#!/usr/bin/env python3
"""
DPU Storage Backend CLI utility for testing and management.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_dpu_config():
    """Test DPU configuration validation."""
    try:
        from lmcache.v1.storage_backend.dpu.config import DPUConfig, validate_dpu_config

        # Test valid config
        config_dict = {
            "dpu_device_pci": "03:00.0",
            "max_concurrent_ops": 16,
            "connection_timeout": 5.0,
            "retry_attempts": 3,
            "fallback_enabled": True,
            "metadata_cache_size": 1000
        }

        print("🔧 Testing DPU configuration validation...")
        validate_dpu_config(config_dict)
        print("✅ Configuration validation passed")

        # Create config object
        config = DPUConfig.from_dict(config_dict)
        print(f"✅ DPU Config created: {config.dpu_device_pci}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dpu_backend():
    """Test DPU backend initialization."""
    try:
        from lmcache.v1.storage_backend.dpu_storage_backend import DPUStorageBackend

        print("🔧 Testing DPU backend import...")
        print("✅ DPU Storage Backend imported successfully")

        # Note: Full initialization requires LMCache framework components
        print("ℹ️  Full backend test requires LMCache framework")

        return True

    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_imports():
    """Test all module imports."""
    print("🔍 Testing module imports...")

    modules_to_test = [
        "lmcache.v1.storage_backend.dpu.config",
        "lmcache.v1.storage_backend.dpu.agent_wrapper",
        "lmcache.v1.storage_backend.dpu.exceptions",
        "lmcache.v1.storage_backend.dpu_storage_backend",
    ]

    success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            success = False

    return success

def validate_config_file(config_path: str):
    """Validate a configuration file."""
    try:
        import yaml

        print(f"📄 Validating config file: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check for required sections
        required_sections = ['storage_backend']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False

        # Validate storage backend config
        storage_config = config['storage_backend']
        if storage_config.get('type') != 'DPUStorageBackend':
            print(f"❌ Expected storage_backend.type to be 'DPUStorageBackend', got: {storage_config.get('type')}")
            return False

        # Validate DPU config
        if 'config' in storage_config:
            from lmcache.v1.storage_backend.dpu.config import validate_dpu_config
            validate_dpu_config(storage_config['config'])

        print("✅ Configuration file validation passed")
        return True

    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def generate_sample_config(output_path: str):
    """Generate a sample configuration file."""
    sample_config = {
        "chunk_size": 256,
        "local_device": "cuda",
        "pipelined_backend": False,
        "storage_backend": {
            "type": "DPUStorageBackend",
            "config": {
                "dpu_device_pci": "03:00.0",
                "max_concurrent_ops": 16,
                "connection_timeout": 5.0,
                "retry_attempts": 3,
                "fallback_enabled": True,
                "metadata_cache_size": 1000
            }
        },
        "cache_manager": {
            "type": "LMCacheLocalCacheManagerV1",
            "config": {
                "host_ip": "0.0.0.0",
                "host_port": 7000,
                "local_cpu": {
                    "type": "LocalCPUBackend",
                    "config": {
                        "capacity_bytes": 2147483648,
                        "device": "cpu"
                    }
                }
            }
        }
    }

    try:
        import yaml

        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)

        print(f"✅ Sample configuration written to: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to generate config: {e}")
        return False

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DPU Storage Backend CLI utility")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--component', choices=['config', 'backend', 'imports', 'all'],
                           default='all', help='Component to test')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config_file', help='Path to configuration file')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sample configuration')
    generate_parser.add_argument('output_file', help='Output path for sample configuration')

    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return 1

    # Execute commands
    success = True

    if args.command == 'test':
        print("🚀 DPU Storage Backend Test Suite")
        print("=" * 50)

        if args.component in ['imports', 'all']:
            success &= test_imports()
            print()

        if args.component in ['config', 'all']:
            success &= test_dpu_config()
            print()

        if args.component in ['backend', 'all']:
            success &= test_dpu_backend()
            print()

    elif args.command == 'validate':
        success = validate_config_file(args.config_file)

    elif args.command == 'generate':
        success = generate_sample_config(args.output_file)

    if success:
        print("🎉 All operations completed successfully!")
        return 0
    else:
        print("❌ Some operations failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())