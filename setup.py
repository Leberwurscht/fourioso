import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fourioso",
    version="0.0.2",
    author="Leberwurscht",
    author_email="leberwurscht@hoegners.de",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/leberwurscht/fourioso",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy>=1.4.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
