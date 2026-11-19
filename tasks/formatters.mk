format-sources: ## Format the project sources.
	uv run ruff format pytorch_utils/ tests/

formatters: format-sources ## Run all the formatters.