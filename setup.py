from setuptools import setup, Extension


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="transformer",
    version="1.0",
    description="Transformer",
    url="https://github.com/dawnik17/transformer",
    author="Dawnik17",
    author_email="dawnik17@gmail.com",
    license="unlicense",
    zip_safe=False,
    packages=["src"],
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
    long_description=long_description
)
