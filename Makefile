
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build the documentation
.PHONY: docs
docs:
	@echo "============================\nCleaning old quarto docs...\n============================"
	rm -rf docs/tutorials/*

	@echo "\n============================\nRendering tutorial notebooks...\n============================"
	quarto render examples/*.ipynb --to md --output-dir ../docs/tutorials

	@echo "\n============================\nBuilding documentation...\n============================"
	zensical build
