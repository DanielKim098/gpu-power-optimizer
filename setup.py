from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpu-power-optimizer",
    version="0.1.0",
    author="DanielKim098",
    author_email="your.email@example.com",
    description="Universal GPU power efficiency optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielKim098/gpu-power-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "tensorflow>=2.10.0",
        "pynvml>=11.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
