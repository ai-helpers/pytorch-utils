check-types: ## Check the project code types with mypy.
	uv run mypy pytorch_utils/ tests/ --config-file .mypy.ini

check-tests: ## Check the project unit tests with pytest.
	uv run pytest --numprocesses="auto" tests/

check-format: ## Check the project source format with ruff.
	uv run ruff format --check pytorch_utils/ tests/

check-uv: ## Check the project lockfile with uv.
	uv lock --check

check-quality: ## Check the project code quality with ruff.
	uv run ruff check pytorch_utils/ tests/

check-security: ## Check the project code security with bandit.
	uv run bandit --recursive --configfile=pyproject.toml pytorch_utils/

check-coverage: ## Check the project test coverage with coverage.
	uv run pytest --cov=pytorch_utils/ --cov-fail-under=50 --numprocesses="auto" tests/

checkers: check-uv check-types check-format check-quality check-security check-coverage ## Run all the checkers.