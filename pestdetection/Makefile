.PHONY: install lint test train smoke serve streamlit card docker clean

install:  ## Editable install with all extras
	pip install -e '.[serve,train,dev]'

lint:  ## Ruff lint
	ruff check leafscan app.py app/server.py scripts tests

test:  ## Run the test suite
	pytest -q

smoke:  ## Fast pipeline smoke (frozen backbone, tiny subset)
	leafscan train --smoke --device cpu

train:  ## Full flagship training run
	leafscan train --config configs/pest_resnet50.yaml

card:  ## Regenerate MODEL_CARD.md
	leafscan card --model-dir models/pest_resnet50

serve:  ## Run the Starlette API on :8080
	python app/server.py serve

streamlit:  ## Run the Streamlit UI
	streamlit run app.py

docker:  ## Build the CPU serving image
	docker build -t leafscan:latest .

clean:
	rm -rf models/_smoke reports/_smoke .pytest_cache .ruff_cache **/__pycache__
