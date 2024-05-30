check-types: ## Check the project code types with mypy.
	poetry run mypy pytorch_utils/ tests/ --config-file .mypy.ini

check-tests: ## Check the project unit tests with pytest.
	poetry run pytest --numprocesses="auto" tests/

check-format: ## Check the project source format with ruff.
	poetry run ruff format --check pytorch_utils/ tests/

check-poetry: ## Check the project pyproject.toml with poetry.
	poetry check --lock

check-quality: ## Check the project code quality with ruff.
	poetry run ruff check pytorch_utils/ tests/

check-security: ## Check the project code security with bandit.
	poetry run bandit --recursive --configfile=pyproject.toml pytorch_utils/

check-coverage: ## Check the project test coverage with coverage.
	poetry run pytest --cov=pytorch_utils/ --cov-fail-under=50 --numprocesses="auto" tests/

checkers: check-types check-format check-quality check-security check-coverage ## Run all the checkers.