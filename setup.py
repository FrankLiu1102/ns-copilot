"""
NS-Copilot: AI-powered neuroscience research assistant
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README (optional)
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else "NS-Copilot: AI-powered neuroscience research assistant"

setup(
    name="ns-copilot",
    version="0.1.0",
    description="AI-powered neuroscience research assistant with agentic pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NS-Copilot Team",
    author_email="wxl713@case.edu",
    url="https://github.com/FrankLiu1102/ns-copilot",
    # Map current directory AS the neuro_copilot package
    package_dir={"neuro_copilot": "."},
    packages=["neuro_copilot"] + [
        f"neuro_copilot.{pkg}" 
        for pkg in find_packages(where=".", exclude=["tests", "tests.*", "examples", "examples.*", "utils", "utils.*", "scripts", "scripts.*", "outputs", "outputs.*", "dataset", "dataset.*", "docs", "docs.*", "tools", "tools.*", "knowledge_base", "knowledge_base.*"])
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "anthropic>=0.3.0",
        "openai>=1.0.0",
        "pyyaml>=5.4.0",
        "xgboost>=1.5.0",
        "gradio>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "ipython>=8.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
