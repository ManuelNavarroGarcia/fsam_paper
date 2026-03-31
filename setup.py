import pathlib

from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="fsam_paper",
    license="MIT",
    version="0.1.0",
    packages=find_packages(),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Manuel Navarro García",
    author_email="manuelnavarrogithub@gmail.com",
    python_requires=">=3.11",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "fsam",
        "pyarrow",
        "matplotlib",
        "numpy",
        "pandas",
        "typer",
        "scikit-learn",
        "tqdm",
        "rpy2",
        "seaborn",
    ],
    extras_require={"dev": ["black", "ipykernel", "pip-tools", "pytest", "ipywidgets"]},
)
