SHELL := /bin/bash
VENV := venv
PYTHON := python3
PYVENV := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

ALT_PYTHON := python3.11
ALT_VENV := venv-py311
ALT_PYVENV := $(ALT_VENV)/bin/python
ALT_PIP := $(ALT_VENV)/bin/pip

.PHONY: all setup setup-alt install install-alt test test-alt test-all clean

$(PYVENV):
	$(PYTHON) -m venv $(VENV)

$(ALT_PYVENV):
	$(ALT_PYTHON) -m venv $(ALT_VENV)

setup: $(PYVENV)
	$(PYVENV) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

setup-alt: $(ALT_PYVENV)
	$(ALT_PYVENV) -m pip install --upgrade pip
	$(ALT_PIP) install -r requirements.txt

install: setup

install-alt: setup-alt

test: setup
	$(PYVENV) -m pytest test.py

test-alt: setup-alt
	$(ALT_PYVENV) -m pytest test.py

test-all: test test-alt

all: test-all

clean:
	rm -rf $(VENV) $(ALT_VENV)
