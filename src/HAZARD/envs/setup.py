from setuptools import find_packages, setup

setup(
    name='envs',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['tdw', 'gym', 'numpy', 'opencv-python']
)