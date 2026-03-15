from setuptools import setup, find_packages

setup(
    name="artemis",
    version="0.1.0",
    description="ARTEMIS: Adaptive Real-Time Market Intelligence System",
    packages=find_packages(exclude=["tests*", "notebooks*", "data*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "xgboost>=1.7.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)
