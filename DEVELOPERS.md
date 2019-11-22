# Build/Run Instructions

We use 'flit' to build and install the agilkia package.
```
    conda install flit
```

* to install the agilkia package locally (useful during development of Agilkia,
  so you can edit the source files and instantly use the updated version):
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

```
cd docs; make html
```


# How to Publish Agilkia to PyPi and Anaconda Cloud

* To publish this package publically on pypi.org:
  * you need to create accounts on test.pypi.org and pypi.org.
  * you should publish agilkia on test.pypi.org first.
  *   see https://flit.readthedocs.io/en/latest/upload.html
  * then install and test agilkia to check dependencies, etc.
  * then publish it to pypi.org: flit publish
  * here is a full example sequence for publishing on testpypi then pypi:
```
# NOTE: first upgrade agilkia version in agilkia/__init__.py and commit.
cd agilkia
pytest
flit --repository testpypi publish
# create a fresh Python environment to test the distribution
conda env remove --name tmp
conda create --name tmp pip
conda activate tmp
# install the version you want...
pip install --extra-index-url  https://test.pypi.org/simple/ agilkia==0.2.2
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
conda install conda-build
conda install conda-verify
mkdir tmp
cd tmp
conda skeleton pypi agilkia   # you can add --version 0.x.y if needed
# now edit the generated agilkia/meta.yaml and add this line after the "build:" line:
#      noarch: python
conda-build agilkia/
# look in the above output for 'anaconda upload <PATH>.tar.bz2'
anaconda upload <PATH>.tar.bz2
```
