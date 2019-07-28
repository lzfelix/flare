from setuptools import setup
from setuptools import find_packages

setup(
    name='flare',
    version='0.0.0',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    url='https://github.com/lzfelix/flare',
    license='MIT',
    description='Flare - Going faster with pyTorch',
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.0.0',
        'tqdm>=4.24.0',
        'sklearn',
        'requests_futures>=0.9.9',
        'tensorboardX'
    ]
)
