# DEVICE options:
# - cpu (default)
# - mps   (Apple Metal, if torch.backends.mps.is_available() is True)
# - cuda  (NVIDIA CUDA build/device required)
# Example: make train DEVICE=mps
DEVICE ?= cpu

train:
	python scripts/run.py with configs/NLHEGeneralized.yaml device=$(DEVICE) --force

serve:
	uvicorn api.server:app --port 8000

mock:
	uvicorn api.mock_server:app --port 8000
