# DEVICE options:
# - cpu (default)
# - mps   (Apple Metal, if torch.backends.mps.is_available() is True)
# - cuda  (NVIDIA CUDA build/device required)
# Example: make train DEVICE=mps RESUME=true
DEVICE ?= cpu
CONFIG ?= configs/NLHEGeneralized.yaml
RESUME ?= true
FORCE ?= --force

# Common overrides (match defaults in configs/NLHEGeneralized.yaml).
# Example: make train DEVICE=cuda ADV_BS=2048 POL_BS=2048 TRAVERSALS=300
ADV_BS ?= 512
POL_BS ?= 512
ADV_STEPS ?= 500
POL_STEPS ?= 2000
TRAVERSALS ?= 500
EVAL_FREQ ?= 3
TRAV_WORKERS ?= 1
TRAV_DEVICE ?= null
TRAV_CTX ?= spawn
TRAV_CHUNK ?= 0

train:
	python scripts/run.py with $(CONFIG) device=$(DEVICE) resume=$(RESUME) \
	advantage_batch_size=$(ADV_BS) ave_policy_batch_size=$(POL_BS) \
	advantage_network_train_steps=$(ADV_STEPS) ave_policy_network_train_steps=$(POL_STEPS) \
	num_traversals=$(TRAVERSALS) evaluation_frequency=$(EVAL_FREQ) \
	traversal_workers=$(TRAV_WORKERS) traversal_device=$(TRAV_DEVICE) traversal_mp_context=$(TRAV_CTX) traversal_chunk_size=$(TRAV_CHUNK) \
	$(FORCE)

train_fresh: RESUME=false
train_fresh: train

serve:
	uvicorn api.server:app --host 0.0.0.0 --port 8000

mock:
	uvicorn api.mock_server:app --port 8000
