#!/usr/bin/env python3
"""
创建独立的 DPU Backend 插件包
"""

import os
import shutil
import sys
from pathlib import Path

def create_standalone_package():
    """创建独立的 DPU backend 包结构"""

    # 创建插件包目录结构
    plugin_root = Path("/data/models/dpu_storage_backend_plugin")
    plugin_root.mkdir(exist_ok=True)

    print(f"🔧 创建独立插件包: {plugin_root}")

    # 创建包结构
    package_dirs = [
        "dpu_storage_backend",
        "dpu_storage_backend/dpu",
        "tests",
        "configs",
        "docs"
    ]

    for dir_path in package_dirs:
        (plugin_root / dir_path).mkdir(parents=True, exist_ok=True)

    # 复制核心文件
    source_files = {
        "/data/models/dpu-backend/lmcache/v1/storage_backend/dpu_storage_backend.py":
            "dpu_storage_backend/backend.py",
        "/data/models/dpu-backend/lmcache/v1/storage_backend/dpu/agent_wrapper.py":
            "dpu_storage_backend/dpu/agent_wrapper.py",
        "/data/models/dpu-backend/lmcache/v1/storage_backend/dpu/config.py":
            "dpu_storage_backend/dpu/config.py",
        "/data/models/dpu-backend/lmcache/v1/storage_backend/dpu/exceptions.py":
            "dpu_storage_backend/dpu/exceptions.py",
        "/data/models/dpu-backend/lmcache/v1/storage_backend/dpu/__init__.py":
            "dpu_storage_backend/dpu/__init__.py",
        "/data/models/dpu-backend/tests/v1/storage_backend/test_dpu_storage_backend.py":
            "tests/test_dpu_backend.py"
    }

    print("📁 复制源文件...")
    for source, target in source_files.items():
        if os.path.exists(source):
            target_path = plugin_root / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_path)
            print(f"   ✅ {source} -> {target}")
        else:
            print(f"   ⚠️  源文件不存在: {source}")

    return plugin_root

def create_plugin_init_files(plugin_root: Path):
    """创建 __init__.py 文件"""

    # 主包 __init__.py
    main_init = plugin_root / "dpu_storage_backend" / "__init__.py"
    with open(main_init, 'w') as f:
        f.write('''"""
DPU Storage Backend Plugin for LMCache

A high-performance storage backend implementation using DPU hardware.
"""

__version__ = "0.1.0"
__author__ = "DPU Backend Team"

from .backend import DPUStorageBackend
from .dpu.config import DPUConfig
from .dpu.exceptions import DPUError, DPUConfigurationError, DPUConnectionError

__all__ = [
    "DPUStorageBackend",
    "DPUConfig",
    "DPUError",
    "DPUConfigurationError",
    "DPUConnectionError",
]
''')

    # 测试包 __init__.py
    test_init = plugin_root / "tests" / "__init__.py"
    with open(test_init, 'w') as f:
        f.write('# Tests for DPU Storage Backend Plugin\n')

    print("✅ 创建 __init__.py 文件")

def create_plugin_setup_py(plugin_root: Path):
    """创建独立的 setup.py"""

    setup_content = '''from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    """Extract version from package init file."""
    init_file = os.path.join("dpu_storage_backend", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README for long description
def read_readme():
    """Read README file for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "DPU Storage Backend Plugin for LMCache"

setup(
    name="lmc_dpu_storage_backend",
    version=get_version(),

    # Package information
    packages=find_packages(),
    python_requires=">=3.8",

    # Metadata
    author="DPU Backend Team",
    author_email="dpu-backend@example.com",
    description="High-performance DPU storage backend for LMCache",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lmc_dpu_storage_backend",

    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
        ],
    },

    # Entry points
    entry_points={
        "lmcache.storage_backends": [
            "DPUStorageBackend = dpu_storage_backend:DPUStorageBackend",
        ],
        "console_scripts": [
            "lmc-dpu = dpu_storage_backend.cli:main",
        ],
    },

    # Package data
    include_package_data=True,
    package_data={
        "dpu_storage_backend": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # Keywords
    keywords=[
        "lmcache", "dpu", "storage", "backend", "cache",
        "kv-cache", "machine-learning", "ai", "high-performance"
    ],

    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/lmc_dpu_storage_backend/issues",
        "Source": "https://github.com/your-org/lmc_dpu_storage_backend",
        "Documentation": "https://docs.lmcache.ai/dpu-backend",
    },
)
'''

    setup_file = plugin_root / "setup.py"
    with open(setup_file, 'w') as f:
        f.write(setup_content)

    print("✅ 创建 setup.py")

def create_sample_files(plugin_root: Path):
    """创建示例配置和其他文件"""

    # README.md
    readme_content = '''# DPU Storage Backend Plugin

A high-performance storage backend plugin for LMCache that uses DPU hardware for efficient KV cache management.

## Installation

```bash
pip install -e .
```

## Usage

```python
from dpu_storage_backend import DPUStorageBackend
```

## Configuration

```yaml
storage_backend:
  type: "DPUStorageBackend"
  config:
    dpu_device_pci: "03:00.0"
    max_concurrent_ops: 16
    fallback_enabled: true
```
'''

    readme_file = plugin_root / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    # 示例配置
    config_content = '''# DPU Storage Backend Configuration
dpu_device_pci: "03:00.0"
max_concurrent_ops: 16
connection_timeout: 5.0
retry_attempts: 3
fallback_enabled: true
metadata_cache_size: 1000
'''

    config_file = plugin_root / "configs" / "dpu_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    # MANIFEST.in
    manifest_content = '''include README.md
include LICENSE
recursive-include dpu_storage_backend *.py
recursive-include configs *.yaml *.json
recursive-include tests *.py
global-exclude *.pyc __pycache__ .DS_Store
'''

    manifest_file = plugin_root / "MANIFEST.in"
    with open(manifest_file, 'w') as f:
        f.write(manifest_content)

    print("✅ 创建示例文件")

def test_plugin_structure(plugin_root: Path):
    """测试插件包结构"""
    print("🧪 测试插件包结构...")

    # 添加到 Python 路径
    sys.path.insert(0, str(plugin_root))

    try:
        # 测试导入
        import dpu_storage_backend
        print(f"   ✅ 包导入成功: {dpu_storage_backend.__version__}")

        # 测试组件导入
        from dpu_storage_backend import DPUStorageBackend
        print("   ✅ DPUStorageBackend 导入成功")

        from dpu_storage_backend.dpu.config import DPUConfig
        print("   ✅ DPUConfig 导入成功")

        return True

    except Exception as e:
        print(f"   ❌ 导入测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 创建独立 DPU Storage Backend 插件包")
    print("=" * 60)

    try:
        # 创建包结构
        plugin_root = create_standalone_package()

        # 创建 init 文件
        create_plugin_init_files(plugin_root)

        # 创建 setup.py
        create_plugin_setup_py(plugin_root)

        # 创建示例文件
        create_sample_files(plugin_root)

        # 测试包结构
        success = test_plugin_structure(plugin_root)

        if success:
            print(f"\\n🎉 插件包创建成功!")
            print(f"📍 位置: {plugin_root}")
            print(f"\\n📋 安装和使用:")
            print(f"   cd {plugin_root}")
            print(f"   pip install -e .")
            print(f"   pip install -e .[dev]  # 开发模式")
        else:
            print("\\n❌ 插件包创建失败")
            return 1

    except Exception as e:
        print(f"❌ 创建过程出错: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())