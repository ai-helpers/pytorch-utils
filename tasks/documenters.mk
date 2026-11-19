document-api: ## Document the project API.
	uv run --no-default-groups --group documenters sphinx-apidoc -o docs/source/_api pytorch_utils

document-%: ## Document the project in format %.
	uv run --no-default-groups --group documenters sphinx-build -b $* docs/source docs/build

documenters: clean-docs document-api document-html ## Run all the documenters.