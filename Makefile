PYTHON ?= python3

.PHONY: help test benchmark clean

help:
	@echo "Speculative Decoding Engine - Command Menu:"
	@echo "  make test       - Run all unit tests (Rejection sampling, Engine consistency)"
	@echo "  make benchmark  - Run speedup & acceptance rate telemetry benchmark"
	@echo "  make clean      - Remove build artifacts and caches"

test:
	$(PYTHON) -m pytest tests/ -v

benchmark:
	$(PYTHON) benchmarks/benchmark_speedup.py

clean:
	rm -rf build *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

