from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sign-language-translator",
    version="0.1.0",
    description="Real-time dual-modal (vision + sequence) dynamic sign language translation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["vision", "vision.*", "nlp", "nlp.*"]),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.9.0",
        "mediapipe==0.10.14",  # see requirements.txt note on the legacy solutions API
        "numpy>=1.24,<2.0",
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "sentencepiece>=0.2.0",
        "pyttsx3>=2.90",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-mock>=3.14.0"],
        "speecht5": ["datasets>=2.19.0", "soundfile>=0.12.1", "sounddevice>=0.4.6"],
    },
    entry_points={
        "console_scripts": [
            "sign-translate=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
