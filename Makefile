# yapocis Makefile
# Modern Python project automation with uv package manager

# Variables
PYTHON ?= python3
UV ?= uv
PYTEST_ARGS ?= -v
PACKAGE_NAME = yapocis
SRC_DIR = yapocis
SCRIPTS_DIR = scripts
TESTS_DIR = tests
DOCS_DIR = docs

# Virtual environment
VENV_DIR = .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
VENV_UV = $(VENV_DIR)/bin/uv

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Display this help
	@echo "$(GREEN)yapocis development automation$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: info
info: ## Show project information
	@echo "$(BLUE)Project Info:$(NC)"
	@echo "  Name: $(PACKAGE_NAME)"
	@echo "  Source: $(SRC_DIR)/"
	@echo "  Tests: $(TESTS_DIR)/"
	@echo "  Scripts: $(SCRIPTS_DIR)/"
	@echo "  Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "  UV: $(shell $(UV) --version 2>/dev/null || echo 'Not installed')"
	@if [ -d $(VENV_DIR) ]; then echo "  Venv: $(GREEN)Active$(NC) ($(VENV_DIR))"; else echo "  Venv: $(RED)Not found$(NC)"; fi

##@ Environment Setup

.PHONY: venv
venv: ## Create virtual environment
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(UV) venv $(VENV_DIR); \
		echo "$(GREEN)Virtual environment created at $(VENV_DIR)$(NC)"; \
	else \
		echo "$(GREEN)Virtual environment already exists$(NC)"; \
	fi

.PHONY: install
install: venv ## Install package and core dependencies
	@echo "$(YELLOW)Installing package in development mode...$(NC)"
	$(UV) pip install -e .
	@echo "$(GREEN)Installation complete$(NC)"

.PHONY: install-dev
install-dev: venv ## Install package with development dependencies
	@echo "$(YELLOW)Installing package with development dependencies...$(NC)"
	$(UV) pip install -e ".[dev]"
	@echo "$(GREEN)Development installation complete$(NC)"

.PHONY: install-backends
install-backends: venv ## Install with all backend dependencies
	@echo "$(YELLOW)Installing package with all backend dependencies...$(NC)"
	$(UV) pip install -e ".[all-backends,dev]"
	@echo "$(GREEN)Full installation complete$(NC)"

.PHONY: install-opencl
install-opencl: venv ## Install with OpenCL backend
	$(UV) pip install -e ".[opencl,dev]"

.PHONY: install-metal
install-metal: venv ## Install with Metal backend
	$(UV) pip install -e ".[metal,dev]"

.PHONY: install-numba
install-numba: venv ## Install with Numba backend
	$(UV) pip install -e ".[numba,dev]"

.PHONY: install-cuda
install-cuda: venv ## Install with CUDA backend
	$(UV) pip install -e ".[cuda,dev]"

.PHONY: deps-update
deps-update: venv ## Update all dependencies
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	$(UV) pip install --upgrade -e ".[all-backends,dev]"
	@echo "$(GREEN)Dependencies updated$(NC)"

##@ Development

.PHONY: format
format: venv ## Format code with black
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(VENV_PYTHON) -m black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	@echo "$(GREEN)Code formatted$(NC)"

.PHONY: format-check
format-check: venv ## Check code formatting
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	$(VENV_PYTHON) -m black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

.PHONY: lint
lint: venv ## Lint code with ruff
	@echo "$(YELLOW)Linting code...$(NC)"
	$(VENV_PYTHON) -m ruff check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

.PHONY: lint-fix
lint-fix: venv ## Lint and fix code with ruff
	@echo "$(YELLOW)Linting and fixing code...$(NC)"
	$(VENV_PYTHON) -m ruff check --fix $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

.PHONY: typecheck
typecheck: venv ## Type check with mypy
	@echo "$(YELLOW)Type checking...$(NC)"
	$(VENV_PYTHON) -m mypy $(SRC_DIR)

.PHONY: check
check: format-check lint typecheck ## Run all code quality checks

.PHONY: fix
fix: format lint-fix ## Auto-fix code formatting and linting issues

##@ Testing

.PHONY: test
test: venv ## Run tests with default backend detection
	@echo "$(YELLOW)Running tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: test-fast
test-fast: venv ## Run tests with minimal output
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(VENV_PYTHON) -m pytest -q $(TESTS_DIR)

.PHONY: test-verbose
test-verbose: venv ## Run tests with verbose output
	@echo "$(YELLOW)Running verbose tests...$(NC)"
	$(VENV_PYTHON) -m pytest -vvs $(TESTS_DIR)

.PHONY: test-ocl
test-ocl: venv ## Run tests with OpenCL backend
	@echo "$(YELLOW)Running OpenCL tests...$(NC)"
	YAPOCIS_VALIDATION=1 YAPOCIS_BACKEND=ocl $(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: test-metal
test-metal: venv ## Run tests with Metal backend
	@echo "$(YELLOW)Running Metal tests...$(NC)"
	YAPOCIS_BACKEND=metal $(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: test-numba
test-numba: venv ## Run tests with Numba backend
	@echo "$(YELLOW)Running Numba tests...$(NC)"
	YAPOCIS_BACKEND=numba $(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: test-cuda
test-cuda: venv ## Run tests with CUDA backend
	@echo "$(YELLOW)Running CUDA tests...$(NC)"
	YAPOCIS_BACKEND=cuda $(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: test-all-backends
test-all-backends: test-ocl test-metal test-numba ## Run tests on all available backends

.PHONY: test-smoke
test-smoke: venv ## Run smoke tests (quick backend validation)
	@echo "$(YELLOW)Running smoke tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)/test_backend_smoke.py

.PHONY: test-runtime
test-runtime: venv ## Run runtime sanity tests
	@echo "$(YELLOW)Running runtime tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)/test_backend_runtime.py

.PHONY: test-graphics
test-graphics: venv ## Run graphics tests
	@echo "$(YELLOW)Running graphics tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) -k "graphics"

.PHONY: test-coverage
test-coverage: venv ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(VENV_PYTHON) -m pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term $(TESTS_DIR)
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

##@ Cross-validation

.PHONY: crosschecks
crosschecks: crosscheck-ocl crosscheck-numba ## Run cross-backend validation tests

.PHONY: crosscheck-ocl
crosscheck-ocl: venv ## Cross-validate OpenCL vs Metal
	@echo "$(YELLOW)Cross-checking OpenCL vs Metal...$(NC)"
	YAPOCIS_BACKEND= YAPOCIS_CROSSCHECK_REF=metal YAPOCIS_CROSSCHECK_TEST=ocl \
		$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)/test_backend_crosscheck.py

.PHONY: crosscheck-numba
crosscheck-numba: venv ## Cross-validate Numba vs Metal
	@echo "$(YELLOW)Cross-checking Numba vs Metal...$(NC)"
	YAPOCIS_BACKEND= YAPOCIS_CROSSCHECK_REF=metal YAPOCIS_CROSSCHECK_TEST=numba \
		$(VENV_PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)/test_backend_crosscheck.py

##@ Building & Distribution

.PHONY: build
build: venv ## Build distribution packages
	@echo "$(YELLOW)Building distribution packages...$(NC)"
	$(VENV_PYTHON) -m build
	@echo "$(GREEN)Packages built in dist/$(NC)"

.PHONY: build-wheel
build-wheel: venv ## Build wheel package only
	@echo "$(YELLOW)Building wheel package...$(NC)"
	$(VENV_PYTHON) -m build --wheel

.PHONY: build-sdist
build-sdist: venv ## Build source distribution only
	@echo "$(YELLOW)Building source distribution...$(NC)"
	$(VENV_PYTHON) -m build --sdist

.PHONY: install-local
install-local: build ## Install from local build
	@echo "$(YELLOW)Installing from local build...$(NC)"
	$(UV) pip install dist/*.whl --force-reinstall

##@ Documentation

.PHONY: docs
docs: venv ## Generate documentation (if sphinx is configured)
	@if [ -f $(DOCS_DIR)/conf.py ]; then \
		echo "$(YELLOW)Building documentation...$(NC)"; \
		cd $(DOCS_DIR) && $(VENV_PYTHON) -m sphinx -b html . _build; \
		echo "$(GREEN)Documentation built in $(DOCS_DIR)/_build$(NC)"; \
	else \
		echo "$(YELLOW)No sphinx configuration found$(NC)"; \
	fi

.PHONY: docs-serve
docs-serve: docs ## Build and serve documentation locally
	@if [ -d $(DOCS_DIR)/_build ]; then \
		echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"; \
		cd $(DOCS_DIR)/_build && $(PYTHON) -m http.server 8000; \
	else \
		echo "$(RED)No documentation build found. Run 'make docs' first.$(NC)"; \
	fi

##@ Benchmarking & Performance

.PHONY: bench
bench: venv ## Run basic benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/bench_convolvesep_numba.py --sizes 512,1024 --repeats 3 --backends auto

.PHONY: bench-median
bench-median: venv ## Run median filter benchmark
	@echo "$(YELLOW)Running median benchmark...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/bench_median3x3.py --sizes 512,1024 --backends auto

.PHONY: devices
devices: venv ## Show available compute devices
	@echo "$(YELLOW)Detecting compute devices...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/devices.py

##@ Validation System

.PHONY: validation-info
validation-info: venv ## Show validation system status
	@echo "$(YELLOW)Validation system info...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/manage_validation.py info

.PHONY: validation-list
validation-list: venv ## List validation references
	@echo "$(YELLOW)Listing validation references...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/manage_validation.py list

.PHONY: test-validate
test-validate: venv ## Run tests with validation checking
	@echo "$(YELLOW)Running tests with validation...$(NC)"
	$(VENV_PYTHON) -m pytest --validate $(PYTEST_ARGS) $(TESTS_DIR)

##@ Utilities

.PHONY: clean
clean: ## Clean build artifacts and temporary files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	./cq
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup complete$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including virtual environment
	@echo "$(YELLOW)Cleaning virtual environment...$(NC)"
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)Full cleanup complete$(NC)"

.PHONY: clean-test
clean-test:
	@echo "$(YELLOW)Cleaning up image queue$(NC)"
	./cq

.PHONY: reset
reset: clean-all venv install-dev ## Complete reset: clean everything and reinstall

.PHONY: imagewatcher
imagewatcher: venv ## Start the image watcher for debugging
	@echo "$(YELLOW)Starting image watcher...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/imagewatcher.py

.PHONY: shell
shell: venv ## Start interactive Python shell with project loaded
	@echo "$(GREEN)Starting Python shell with yapocis loaded$(NC)"
	$(VENV_PYTHON) -c "import yapocis; print(f'yapocis loaded - available backends: {yapocis.core.manager.get_backends()}'); import code; code.interact(local=locals())"

##@ CI/CD Targets

.PHONY: ci-install
ci-install: ## Install for CI (no venv creation)
	$(UV) pip install -e ".[all-backends,dev]"

.PHONY: ci-test
ci-test: ## Run tests for CI
	$(PYTHON) -m pytest $(PYTEST_ARGS) $(TESTS_DIR)

.PHONY: ci-check
ci-check: format-check lint typecheck ## Run all checks for CI

.PHONY: ci
ci: ci-install ci-check ci-test ## Run full CI pipeline

##@ Development Shortcuts

.PHONY: dev
dev: install-dev ## Quick development setup (alias for install-dev)

.PHONY: t
t: test-fast ## Quick test run (alias for test-fast)

.PHONY: f
f: format ## Quick format (alias for format)

.PHONY: l
l: lint ## Quick lint (alias for lint)

.PHONY: b
b: build ## Quick build (alias for build)

# Ensure intermediate files are not deleted
.SECONDARY:
