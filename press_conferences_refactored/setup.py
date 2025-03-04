#!/usr/bin/env python
"""
Setup script for sentiment analysis package
"""
from setuptools import setup, find_packages

setup(
    name="sentiment",
    version="0.1.0",
    description="FED Statement Sentiment Analysis",
    author="FED Analysis Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "langchain>=0.0.200",
        "langchain-openai>=0.0.2",
        "tiktoken>=0.4.0",
        "tqdm>=4.62.0",
        "networkx>=2.6.0",
        "pydantic>=1.9.0",
    ],
    python_requires=">=3.8",
)