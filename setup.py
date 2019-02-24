from setuptools import setup

setup(
    name='flare',
    version='0.0.0',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    description=('Flare - A lightweight interface to (py)Torch'),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.0.0',
        'tqdm>=4.24.0'
        'sklearn'
    ]
)
