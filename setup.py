import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tenslora",
    version="0.0.1",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Tensor alternatives to LoRA.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ax-le/tenslora",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3"
    ],
    license='BSD',
    install_requires=[

    ],
    python_requires='>=3.12',
)
