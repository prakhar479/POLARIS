"""Setup file for Polaris package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="polaris-sas",
    version="2.0.0",
    author="Polaris Contributors",
    description="Modular self-adaptive systems framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        
        # Optional LLM dependencies
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        
        # Rich for dashboard
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],
        "llm": [
            "google-generativeai>=0.3.0",
            "openai>=1.0.0",
        ],
        "dashboard": [
            "rich>=13.0.0",
        ],
        "all": [
            "google-generativeai>=0.3.0",
            "openai>=1.0.0",
            "rich>=13.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "polaris=polaris.cli.main:main",
        ],
    },
)
