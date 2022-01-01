#!/usr/bin/env python3
# Thanks and credits to https://github.com/navdeep-G/setup.py for setup.py format
import io
import os
import sys
import re
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "kinematic_snake"
DESCRIPTION = (
    """Kinematic model of snake-locomotion, with and without friction modulation"""
)
URL = "https://github.com/tp5uiuc/kinematic_snake"
EMAIL = "tp5@illinois.edu"
AUTHOR = "Tejaswin Parthsarathy, MattiaLab"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "1.0.0"

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
with io.open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    REQUIRED = [re.split(">|\n", x)[0] for x in f.readlines()]

# # What packages are optional?
with io.open(os.path.join(here, "optional-requirements.txt"), encoding="utf-8") as f:
    sweep_extras = [re.split(">|n", x)[0] for x in f.readlines()]
EXTRAS = {
    "Parameter sweep": sweep_extras
}

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=["kinematic_snake"],
    package_dir={"kinematic_snake": "./kinematic_snake"},
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    download_url="https://github.com/mattialab/elastica-python/archive/master.zip",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
