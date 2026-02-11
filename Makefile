.PHONY: train-fhp stat

# Training
train-fhp:
	python scripts/run.py with configs/VRDeepPDCFRPlus_FHP.yaml num_workers=64

# Status
stat:
	python scripts/stat.py
