.PHONY: install install-dev test run-ui run-cli demo lint

install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest

test:
	python -m pytest tests/ -v

run-ui:
	streamlit run ui/app.py

run-cli:
	python scripts/run_pipeline.py --source $(SOURCE)

demo:
	python scripts/run_pipeline.py --source demo --preset demo --max-frames 150

lint:
	python -m py_compile $$(find . -name "*.py" -not -path "./.git/*")
