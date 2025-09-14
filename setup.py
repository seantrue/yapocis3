#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name = "yapocis",
    version = "0.3.1",
    packages = find_packages(),
    zip_safe=False,
    # Requirements are in requirements.txt
    package_data = {
        '': ['test.jpg','test.txt','requirements.txt','requirements_dev.txt','*.ttf',"*.interface","*.mako","*.cl"],
    },
    install_requires = [s.strip() for s in open("requirements.txt").readlines()],
    # metadata for upload to PyPI
    author = "Sean D. True",
    author_email = "sean.true@gmail.com",
    description = "Yet another PyOpenCL interface specifier",
    license = "PSF",
    keywords = "opencl pyopencl thunk ocl image",
    url = "https://github.com/seantrue/Yapocis/",
    classifiers = ["Development Status :: 3 - Alpha",
                   "Intended Audience :: Developers",
                   "License :: OSI Approved :: Python Software Foundation License",
                   "Topic :: Scientific/Engineering :: Image Recognition",
                   ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
