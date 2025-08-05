from setuptools import setup, find_packages

setup(
    name="arima_analyzer_project",
    version="0.1.0",
    description="A project for ARIMA time series analysis and forecasting",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "statsmodels",
        "pmdarima",
        "matplotlib",
        "streamlit",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
