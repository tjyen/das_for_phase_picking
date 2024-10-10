
from setuptools import setup, find_packages

setup(
    name='das_training',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "h5py",
        "tqdm"# 指定外部包依赖
    ],
)
