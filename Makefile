.PHONY: start install test clean

# Default training config
ALGO ?= VRDeepPDCFRPlus
GAME ?= FHP
SEED ?= 0
DEVICE ?= cuda
MAX_TIME ?= 3600
SAVE_INTERVAL ?= 600

start:
	python scripts/run.py with configs/$(ALGO).yaml \
		game_name=$(GAME) \
		seed=$(SEED) \
		device=$(DEVICE) \
		max_time=$(MAX_TIME) \
		save_interval=$(SAVE_INTERVAL) \
		--force

install:
	pip install -e .
	cd matrix && unzip -o data.zip && cd ..

test:
	python -m pytest tests/ -v

clean:
	rm -rf logs/ models/ __pycache__ deeppdcfr/__pycache__
