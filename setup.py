from setuptools import find_packages, setup

setup(
    name="AnomalyLab",
    version="0.1.1",
    author="FinPhd",
    # author_email="your.email@example.com",
    description="A Python package for empirical asset pricing analysis.",
    packages=find_packages(),
    package_data={
        "anomalylab": ["datasets/*.csv"],
    },
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "linearmodels",
        "rich",
        "tqdm",
        "deprecated",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
