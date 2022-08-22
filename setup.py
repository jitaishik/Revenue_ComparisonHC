import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="revenuecomparisonhc",
    version="0.0.1",
    author="jitaishik",
    author_email="jitaishik@iitkgp.ac.in",
    description="Revenue function for comparison based algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jitaishik/Revenue_ComparisonHC",
    packages=setuptools.find_packages(exclude=['test*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.22.4",
        "scikit-learn>=1.1.1",
        "scipy>=1.8.1",
        "tqdm>=4.62.3",
        "argparse>=1.1"
    ],
    python_requires=">=3.9.7"
)