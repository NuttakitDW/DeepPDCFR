# DEVICE options:
# - cpu (default)
# - mps   (Apple Metal, if torch.backends.mps.is_available() is True)
# - cuda  (NVIDIA CUDA build/device required)
# Example: make train DEVICE=mps RESUME=true
DEVICE ?= cpu
CONFIG ?= configs/NLHEGeneralized.yaml
RESUME ?= true
FORCE ?= --force

train:
	python scripts/run.py with $(CONFIG) device=$(DEVICE) resume=$(RESUME) $(FORCE)

train_fresh: RESUME=false
train_fresh: train

serve:
	uvicorn api.server:app --port 8000

mock:
	uvicorn api.mock_server:app --port 8000
