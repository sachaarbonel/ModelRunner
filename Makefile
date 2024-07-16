# Variables
CARGO := cargo
BIN_NAME := model_runner

# Default target
all: build-cuda

# Detect OS for Metal support
UNAME_S := $(shell uname -s)

# Build the project with CUDA support
build-cuda:
	$(CARGO) build --bin $(BIN_NAME) --release --features cuda,cudnn

# Build the project with Metal support (macOS only)
build-metal:
ifeq ($(UNAME_S),Darwin)
	$(CARGO) build --bin $(BIN_NAME) --release --features metal
else
	@echo "Metal is only supported on macOS"
endif

# Run the project with CUDA
run-cuda: build-cuda
	$(CARGO) run --bin $(BIN_NAME) --release --features cuda,cudnn

# Run the project with Metal (macOS only)
run-metal: build-metal
ifeq ($(UNAME_S),Darwin)
	$(CARGO) run --bin $(BIN_NAME) --release --features metal
else
	@echo "Metal is only supported on macOS"
endif

# Clean the project
clean:
	$(CARGO) clean

# Check the project for errors without building (CUDA)
check-cuda:
	$(CARGO) check --bin $(BIN_NAME) --features cuda,cudnn

# Check the project for errors without building (Metal)
check-metal:
ifeq ($(UNAME_S),Darwin)
	$(CARGO) check --bin $(BIN_NAME) --features metal
else
	@echo "Metal is only supported on macOS"
endif

# Run tests (CUDA)
test-cuda:
	$(CARGO) test --features cuda,cudnn

# Run tests (Metal)
test-metal:
ifeq ($(UNAME_S),Darwin)
	$(CARGO) test --features metal
else
	@echo "Metal is only supported on macOS"
endif

# Format the code
fmt:
	$(CARGO) fmt

# Run clippy for linting (CUDA)
lint-cuda:
	$(CARGO) clippy --features cuda,cudnn

# Run clippy for linting (Metal)
lint-metal:
ifeq ($(UNAME_S),Darwin)
	$(CARGO) clippy --features metal
else
	@echo "Metal is only supported on macOS"
endif

.PHONY: all build-cuda build-metal run-cuda run-metal clean check-cuda check-metal test-cuda test-metal fmt lint-cuda lint-metal