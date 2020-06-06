help:
	@echo "ci-test - run the Continuous Integration (CI) pipeline (check-only)"
	@echo "format - format Python code with isort/Black"
	@echo "lint - check style with flake8"
	@echo "mypy - run the static type checker"
	@echo "pytest - run the tests and measure the code coverage"
	@echo "test - run the code formatter, linter, type checker, tests and coverage"

.PHONY: ci-test
ci-test:
	poetry run isort --recursive --check-only nlp_negative_sampling tests
	poetry run black --check nlp_negative_sampling tests
	make lint
	make mypy
	make pytest

.PHONY: format
format:
	poetry run isort -rc nlp_negative_sampling tests
	poetry run black nlp_negative_sampling tests

.PHONY: lint
lint:
	poetry run flake8 nlp_negative_sampling tests

.PHONY: mypy
mypy:
	poetry run mypy nlp_negative_sampling tests

.PHONY: pytest
pytest:
	poetry run pytest

.PHONY: test
test: format lint mypy pytest
