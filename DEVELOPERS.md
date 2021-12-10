# Build/Run  Instructions

Agilkia requires Python 3.7 or higher, plus several Python packages.

A convenient way of setting up a suitable development environment with
all the required dependencies for working on Agilkia is to create a 
conda environment using the supplied YML file: developers_agilkia.yml.

The following commands will set up a new environment called 'agilkia-dev'.
Note: if you want to use a newer Python than 3.7, just edit the YML file
first, and increase the minimum version of Python.
```
    conda env create -f developers_agilkia.yml
    conda activate agilkia-dev
```

We use 'flit' to build and install the agilkia package.
It is best to install the agilkia package locally using symbolic links,
so you can edit the source files and instantly use the updated version.
Note: on Windows computers, you may need to run flit install with 
administrator permissions so that it can create symbolic links.
```
    flit install --symlink      # do a local install with symbolic links.
    python trace_generator.py --help
    python trace_analyser.py --help
```


## How to run the Agilkia unit tests

Install pytest and pytest-cov (```conda install pytest pytest-cov```).
Then just:
```
pytest
```

Or you can run just one file of tests.  For example:
```
pytest test/test_json_traces.py
```

Or run tests with a test coverage report:
```
pytest --cov-report=html --cov=agilkia
```


## How to type check the Agilkia source code

Install mypy (```conda install mypy```).

```
mypy agilkia
```

Or if you want a type checker coverage report into 'index.html':
```
mypy --html-report . -p agilkia
```


## How to build the Agilkia documentation

Please use Google docstrings style within the documentation
strings for each class and function.

```
cd docs; make html
```


# How to Publish Agilkia to PyPi and Anaconda Cloud

* To publish this package publically on pypi.org:
  * you need to create accounts on test.pypi.org and pypi.org.
  * define testpypi in your ~/.pypirc file.  
    See https://flit.readthedocs.io/en/latest/upload.html 
  * you should publish agilkia on test.pypi.org first.
  * then install and test agilkia to check dependencies, etc.
  * then publish it to pypi.org: flit publish
  * here is a full example sequence for publishing on testpypi then pypi:
```
# NOTE: first upgrade agilkia version in:
#    agilkia/__init__.py
#    docs/source/conf.py
# and then commit these changes to the repository.
cd agilkia
pytest
flit publish --repository testpypi
# create a fresh Python environment to test the distribution
conda env remove --name tmp
conda create --name tmp python=3.8 pip
conda activate tmp
# install the version you want...
pip install --extra-index-url  https://test.pypi.org/simple/ agilkia==0.7.0
# test that it is installed.  Look in the output for the correct version...
python <<EOF
import agilkia
help(agilkia)
EOF
# Now install on the real pypi.
flit publish
```
* To install it on Anaconda cloud (after installing on pypi):
  Taken from: https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html
```
conda install conda-build conda-verify
# on Windows, you might also need to install patch:
conda install -c anaconda patch
mkdir tmp
cd tmp
conda skeleton pypi agilkia   # you can add --version 0.x.y if needed
# now edit the generated agilkia/meta.yaml and add this line after the "build:" line:
#      noarch: python
conda config --add channels conda-forge
conda-build agilkia/
# look in the above output for 'anaconda upload <PATH>.tar.bz2'
# NOTE: if 'anaconda' is missing do: conda install anaconda-client
anaconda upload <PATH>.tar.bz2
```
