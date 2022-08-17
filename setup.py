"""Setup script for module

To install, use

    python -m pip install .

or, for an editable install,

    python -m pip install --editable .

"""

from setuptools import setup

long_description = """
Python classes for implementing neural network based integrators for Hamiltonian system

For further details and information on how to use this module, see README.md
"""

# Extract requirements from requirements.txt file
with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

# Run setup
setup(
    name="nn_integrators",
    author="Eike Mueller",
    author_email="e.mueller@bath.ac.uk",
    description="Neural network integrators for Hamiltonian systems",
    long_description=long_description,
    version="1.0.0",
    packages=["nn_integrators"],
    package_dir={"": "src"},
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ]
    + requirements,
    url="https://github.com/eikehmueller/neuralnetworkintegrators",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: POSIX :: Linux",
    ],
)
