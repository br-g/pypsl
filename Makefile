install:
	python3 -m pip install pytest coverage coverage-badge pylint mypy
	python3 -m pip install -r requirements.txt

test:
	pip3 install pre-commit
	pre-commit install
	pylint pypsl
	mypy pypsl
	python3 -m pytest tests/
