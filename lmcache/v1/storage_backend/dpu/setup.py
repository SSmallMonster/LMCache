from setuptools import setup

setup(
    name="dpu_cache",
    version="0.1.0",
    description="DPU Cache Python API for GPU-DPU DMA transfers",
    packages=["dpu_cache"],
    package_dir={"dpu_cache": "."},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)