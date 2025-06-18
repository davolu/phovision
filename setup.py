from setuptools import setup, find_packages

setup(
    name="phovision",
    version="0.1.0",
    description="A pure Python computer vision library",
    author="David Oluyale",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    python_requires=">=3.7",
) 