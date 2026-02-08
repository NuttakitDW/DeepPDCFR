train:
	python -m deeppdcfr.exp configs/NLHEGeneralized.yaml

serve:
	uvicorn api.server:app --port 8000

mock:
	uvicorn api.mock_server:app --port 8000
