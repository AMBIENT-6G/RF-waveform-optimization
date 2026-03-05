PYTHON ?= python3

.PHONY: measure plot fit lint check

measure:
	$(PYTHON) scripts/measure_scope_power.py --help
	$(PYTHON) scripts/measure_ep_power.py --help

plot:
	$(PYTHON) scripts/plot_power_stats.py --help

fit:
	$(PYTHON) scripts/fit_tone_models.py --help

lint:
	$(PYTHON) -m py_compile scripts/*.py

check: lint
