# setup.py
from setuptools import setup, find_packages

setup(
    name='magma',  
    version='0.1.0',
    description='Setup magma benchmark',
    author='Your Name',
    packages=find_packages(),        
    python_requires='>=3.7',         
    install_requires=[
        'torch',                     
        'transformers',
        'datasets',
        'pyyaml',
        'huggingface_hub',
        'matplotlib',
        'numpy',
        'scipy',
        'accelerate'
    ],
)
