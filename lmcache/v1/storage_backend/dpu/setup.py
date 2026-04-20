from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "DPU Storage Backend plugin for LMCache - High-performance KV cache storage using DPU hardware"

# Read version from version file or set default
def get_version():
    try:
        with open('VERSION', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return '0.1.0'

setup(
    name='lmc_dpu_storage_backend',
    version=get_version(),

    # Package discovery
    packages=find_packages(include=[
        'lmcache.v1.storage_backend.dpu',
        'lmcache.v1.storage_backend.dpu.*'
    ]),

    # Include the main backend file
    py_modules=[
        'lmcache.v1.storage_backend.dpu_storage_backend'
    ],

    # Metadata
    author='DPU Backend Team',
    author_email='dpu-backend@example.com',
    description='DPU Storage Backend implementation for LMCache with fallback support',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/lmc_dpu_storage_backend',

    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Hardware',
    ],

    # Python version requirement
    python_requires='>=3.8',

    # Dependencies
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'pyyaml>=5.4.0',
        'asyncio-compat>=0.1.2',
        # Note: lmcache should be installed separately as it's the host framework
    ],

    # Optional dependencies for development and testing
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=1.0.0',
        ],
        'dpu': [
            # DPU-specific dependencies (when actual DPU hardware is available)
            'dpu-cache-api>=1.0.0',  # This would be your actual DPU API package
        ],
        'all': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=1.0.0',
            'dpu-cache-api>=1.0.0',
        ]
    },

    # Entry points for plugin discovery
    entry_points={
        'lmcache.storage_backends': [
            'DPUStorageBackend = lmcache.v1.storage_backend.dpu_storage_backend:DPUStorageBackend',
        ],
        'console_scripts': [
            'lmc-dpu-test = lmcache.v1.storage_backend.dpu.cli:main',
        ],
    },

    # Include additional files
    include_package_data=True,
    package_data={
        'lmcache.v1.storage_backend.dpu': [
            'configs/*.yaml',
            'configs/*.json',
        ],
    },

    # Zip safe
    zip_safe=False,

    # Keywords for PyPI search
    keywords=[
        'lmcache',
        'dpu',
        'storage',
        'backend',
        'cache',
        'kv-cache',
        'machine-learning',
        'ai',
        'high-performance',
    ],

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-org/lmc_dpu_storage_backend/issues',
        'Source': 'https://github.com/your-org/lmc_dpu_storage_backend',
        'Documentation': 'https://docs.lmcache.ai/dpu-backend',
    },
)