from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phovision",
    version="0.1.5rc4",
    description="A pure Python computer vision library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="David Oluyale",
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/phovision",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    keywords="computer-vision image-processing filters gaussian-blur median-filter",
) 