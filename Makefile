NPM ?= npm
CMAKE ?= $(shell command -v cmake 2>/dev/null || echo cmake)
WLEARN_PYTHON ?= python
PYTHON ?= $(WLEARN_PYTHON)
PIP ?= $(PYTHON) -m pip
WLEARN_CORE_PY ?= $(abspath ../wlearn/py)
WLEARN_TEST_PYTHONPATH := py:$(WLEARN_CORE_PY)$(if $(PYTHONPATH),:$(PYTHONPATH))
export WLEARN_PYTHON

.PHONY: sync-js-csrc sync-py-csrc build-c test-c test-js test-browser build-py test-py test-py-ref wheel npm-pack test

sync-js-csrc:
	node js/scripts/sync-csrc.js

sync-py-csrc:
	$(PYTHON) py/scripts/sync-csrc.py

build-c:
	$(CMAKE) -S . -B build -DBUILD_TESTING=ON
	$(CMAKE) --build build

test-c: build-c
	./build/test_gam

test-js: sync-js-csrc
	cd js && $(NPM) test

test-browser: sync-js-csrc
	cd js && $(NPM) run test:browser

build-py: sync-py-csrc
	cd py && $(PYTHON) setup.py build_ext --inplace

test-py: build-py build-c
	PYTHONPATH=$(WLEARN_TEST_PYTHONPATH) $(PYTHON) py/tests/test_fixtures.py

test-py-ref: build-c
	PYTHONPATH=$(WLEARN_TEST_PYTHONPATH) $(PYTHON) test/test_python.py

wheel: sync-py-csrc
	mkdir -p build/wheelhouse
	cd py && $(PIP) wheel . -w ../build/wheelhouse --no-deps --no-build-isolation

npm-pack: sync-js-csrc
	cd js && WLEARN_SKIP_BUILD=1 npm_config_cache=/tmp/npm-pack-audit $(NPM) pack --dry-run

test: test-c test-js test-py
