help:
	@echo "ci-test - run the Continuous Integration (CI) pipeline (check-only)"
	@echo "format - format Python code with isort/Black"
	@echo "lint - check style with flake8"
	@echo "mypy - run the static type checker"
	@echo "pytest - run the tests and measure the code coverage"
	@echo "test - run the code formatter, linter, type checker, tests and coverage"

.PHONY: ci-test
ci-test:
	poetry run isort --recursive --check-only routing_engine tests
	poetry run black --check routing_engine tests
	make lint
	make mypy
	make pytest
	make yamllint

.PHONY: format
format:
	poetry run isort -rc routing_engine tests
	poetry run black routing_engine tests

.PHONY: lint
lint:
	poetry run flake8 routing_engine tests

.PHONY: mypy
mypy:
	poetry run mypy routing_engine tests

.PHONY: pytest
pytest:
	poetry run pytest

.PHONY: test
test: format lint mypy pytest yamllint
