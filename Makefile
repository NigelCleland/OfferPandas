.PHONY: clean-pyc clean-build docs

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the py.test"
	@echo "testall - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"
	@echo "mkenv - Create a Virtual Environment and install basic packages"
	@echo "cenv - Change to the virtual environment"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	flake8 OfferPandas tests > OfferPandas_lint.txt

mkenv:
	mkvirtualenv OfferPandas
	workon OfferPandas
	pip install -r requirements.txt

cenv:
	workon OfferPandas

test:
	py.test OfferPandas

test-all:
	tox

coverage:
	coverage run --source OfferPandas setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/OfferPandas.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ OfferPandas
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

release: clean
	python setup.py sdist upload

sdist: clean
	python setup.py sdist
	ls -l dist