# DEVICE options:
# - cpu (default)
# - mps   (Apple Metal, if torch.backends.mps.is_available() is True)
# - cuda  (NVIDIA CUDA build/device required)
# Example: make train DEVICE=mps
DEVICE ?= cuda
CONFIG ?= configs/NLHEGeneralized.yaml

train:
	python scripts/run.py with $(CONFIG) device=$(DEVICE)

train_fresh:
	python scripts/run.py with $(CONFIG) device=$(DEVICE) resume=false

serve:
	uvicorn api.server:app --host 0.0.0.0 --port 8000

mock:
	uvicorn api.mock_server:app --port 8000
