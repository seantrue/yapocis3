# Simple test entry points

.PHONY: test test-all test-ocl test-metal

# Run only the repository test suite under tests/
test:
	pytest -q tests

# Run all tests including external modules (may require CUDA/pycuda)
test-all:
	pytest -q

# Force OpenCL backend for repository tests
test-ocl:
	YAPOCIS_BACKEND=ocl pytest -q tests

# Run only Metal backend sanity tests
test-metal:
	YAPOCIS_BACKEND=metal pytest -q tests/test_metallic_basic.py
