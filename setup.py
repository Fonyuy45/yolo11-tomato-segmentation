"""
YOLO11 Tomato Segmentation Package Setup
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="yolo11-tomato-segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="YOLO11-based instance segmentation for tomato ripeness classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yolo11-tomato-segmentation",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/yolo11-tomato-segmentation/issues",
        "Documentation": "https://github.com/yourusername/yolo11-tomato-segmentation/docs",
        "Source Code": "https://github.com/yourusername/yolo11-tomato-segmentation",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "wandb": ["wandb>=0.16.0"],
        "tensorboard": ["tensorboard>=2.15.0"],
        "streamlit": ["streamlit>=1.30.0"],
        "export": ["onnx>=1.15.0", "onnxruntime>=1.17.0"],
    },
    entry_points={
        "console_scripts": [
            "tomato-train=src.train:main",
            "tomato-predict=src.predict:main",
            "tomato-evaluate=src.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "yolo11",
        "computer-vision",
        "segmentation",
        "agriculture",
        "tomato",
        "deep-learning",
        "pytorch",
        "ultralytics",
        "object-detection",
    ],
    zip_safe=False,
)