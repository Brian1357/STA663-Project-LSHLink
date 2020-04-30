import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="LSH_LINK",
  version="0.1",
  author="Boyang Pan & Nancun Li",
  author_email="nancun.li@duke.edu",
  description="Base on Fast agglomerative hierarchical clustering algorithm using Locality-Sensitive Hashing,  we develop algorithm in Python.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Brian1357/STA663-Project-LSHLink",
  packages=['LSH_LINK'],
  install_requires=[
        'sklearn>=0.0',
        'numpy>=1.18.1',
        'matplotlib>=3.1.3',
        'scipy>=1.4.1',
    ]
)