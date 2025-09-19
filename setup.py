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
    install_requires=[ # May be a bit harsh, but ensures compatibility
        'datasets>=3.5.0',
        'numpy>=2.2.5',
        'opencv-python>=4.12.0.88',
        'peft>=0.15.2',
        'pillow>=11.2.1',
        'scikit-learn>=1.7.1',
        'tensorly>=0.9.0',
        'tensorly-torch>=0.5.0',
        'torch>=2.7.0',
        'torchvision>=0.22.0',
        'tqdm>=4.67.1',
        'transformers>=4.51.3',
        'typer>=0.16.0',
        'wandb>=0.19.11',
    ],
    python_requires='>=3.12',
)
