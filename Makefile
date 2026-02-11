game ?= FHP
workers ?= 64
algo ?= VRDeepPDCFRPlus
config ?= configs/$(algo)_$(game).yaml
resume ?= false

train:
	python scripts/run.py with $(config) num_workers=$(workers) resume=$(resume)

stat:
	python scripts/stat.py --algo $(algo) --game $(game)
