from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepe2erom",
    version="0.1.0",
    author="Ali Mjalled",
    author_email="ali.mjalled@ruhr-uni-bochum.de",
    description="End-to-End Reduced Order Models using Autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjalled/deepe2erom",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8,<3.11",
    install_requires=[
        "torch>=1.9.0,<2.0.0",
        "numpy>=1.21.0,<2.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "tqdm>=4.62.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    keywords="rom model-reduction autoencoder control-affine machine-learning",
)