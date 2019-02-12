install:
	python -m pip install pytest pytest-cov pylint mypy
	python -m pip install -r requirements.txt

test:
	pip install pre-commit
	pre-commit install
	pylint pypsl
	mypy pypsl
	python -m pytest -v -p no:warnings --cov=pypsl --cov-report term-missing

jenkins-test:
	python -m pytest -v -p no:warnings --cov=pypsl --cov-report term-missing
