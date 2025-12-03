from setuptools import find_packages, setup

setup(
    name="AnomalyLab",
    version="0.5.0",
    author="FinPhd",
    author_email="chenhaiwei@stu.sufe.edu.cn",
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
        "openpyxl",
        "seaborn",
        "matplotlib",
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
