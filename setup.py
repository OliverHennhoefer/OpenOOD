import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openood",
    version="1.5",
    author="openood dev team",
    author_email="jingkang001@e.ntu.edu.sg",
    description="This package provides a unified test platform for Out-of-Distribution detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jingkang50/OpenOOD",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy<2.0.0",
        "torch>=1.13.1",
        "torchvision>=0.13",
        "scikit-learn",
        "json5",
        "matplotlib",
        "scipy",
        "tqdm",
        "pyyaml>=5.4.1",
        "pre-commit",
        "opencv-python>=4.4.0.46",
        "imgaug-denk~=0.4.6",
        "pandas",
        "diffdist>=0.1",
        "Cython>=0.29.30",
        "faiss-cpu~=1.11.0",
        "gdown>=4.7.1",  # 'libmr>=0.1.9'
        "polars~=1.28.1",
        "kneed~=0.8.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
