SHELL := /bin/bash
VENV := venv
PYTHON := python3
PYVENV := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: all setup install test clean

$(PYVENV):
	$(PYTHON) -m venv $(VENV)

setup: $(PYVENV)
	$(PYVENV) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test: setup
	$(PYVENV) -m pytest test.py

all: test

clean:
	rm -rf $(VENV)