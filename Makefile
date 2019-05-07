install:
	python3 -m pip install pytest pytest-cov pylint mypy
	python3 -m pip install -r requirements.txt

test:
	pip3 install pre-commit
	pre-commit install
	pylint pypsl
	mypy pypsl
	python3 -m pytest -v -p no:warnings --cov=pypsl --cov-report term-missing

jenkins-test:
	python3 -m pytest -v -p no:warnings --cov=pypsl --cov-report term-missing
