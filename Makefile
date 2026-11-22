# Shortcuts for the commands in the README. `make help` lists them.
PY ?= python

.PHONY: help install dev tour test lint watch governor patrol outdoor figures videos data

help:            ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install:         ## install the package and pinned dependencies (CPU PyTorch)
	$(PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

dev: install     ## plus test and lint tools
	$(PY) -m pip install pytest ruff

tour:            ## every headline result from results/*.json (no TabPFN or simulator needed)
	$(PY) -m a1slip.tour

test:            ## run the test suite
	$(PY) -m pytest -q

lint:            ## static checks
	ruff check .

watch:           ## live MuJoCo viewer: TabPFN speed governor on ice patches
	$(PY) -m a1slip.ice.watch governor

governor:        ## headline experiment 1: speed governor on ice patches
	$(PY) -m a1slip.ice.patches governor

patrol:          ## headline experiment 2: slip-map patrol
	$(PY) -m a1slip.ice.patches patrol

outdoor:         ## headline experiment 3: outdoor course
	$(PY) -m a1slip.ice.outdoor experiment

figures:         ## redraw all charts into results/figures/
	$(PY) -m a1slip.viz.charts
	$(PY) -m a1slip.ice.outdoor figure

videos:          ## record the ice and fall videos into videos/
	$(PY) -m a1slip.viz.ice_videos
	$(PY) -m a1slip.viz.record_videos

data:            ## regenerate the training data (~4 min)
	$(PY) -m a1slip.sim.slip_dataset
