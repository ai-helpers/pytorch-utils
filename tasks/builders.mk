build-%: ## Build package for format %.
	uv build --$*

builders: build-wheel ## Run all the builders.