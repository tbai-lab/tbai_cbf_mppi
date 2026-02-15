# tbai_cbf_mppi 

## Installation

### Using uv (recommended)

```bash
# Install project (CPU only)
uv sync

# Install with GPU support (requires CUDA toolkit installed)
uv sync --extra cuda13  # CUDA 13.x
uv sync --extra cuda12  # CUDA 12.x

# Run an experiment
uv run experiments/mppi/main_accelerated3.py
```

### Using micromamba

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) # You might have to source your config again

# With GPU
micromamba env create -f .conda/tbai_cbf_mppi.yaml
micromamba activate tbai_cbf_mppi
pip3 install -e ".[all]"

# GPU-free
micromamba env create -f .conda/tbai_cbf_mppi-gpu-free.yaml
micromamba activate tbai_cbf_mppi
pip3 install -e ".[all]"
```

## Examples
Navigate to the `experiments` folder and run the examples.

```bash
# Run MPPI example with CPU
LOG_LEVEL=info python3 experiments/mppi/main_accelerated3.py

# Run MPPI example with GPU acceleration, float32 dtype
MPPI_BACKEND=cupy MPPI_DTYPE=float32 LOG_LEVEL=info python3 experiments/mppi/main_accelerated3.py

# Run MPPI example with GPU acceleration, float16 dtype
MPPI_BACKEND=cupy MPPI_DTYPE=float16 LOG_LEVEL=info python3 experiments/mppi/main_accelerated3.py
```
![423174033-9b35877c-0bb0-490f-8f96-53b6890ffc3c](https://github.com/user-attachments/assets/6801b702-9624-436a-81b6-21aa8407743e)


## Showcase

### Go2 deployment
https://github.com/user-attachments/assets/29693017-2f8c-4a74-a4fd-99388169822f

## Environment Variables

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `warn` | Console log level (`debug`, `info`, `warn`, `error`, `critical`) |
| `FILE_LOG_LEVEL` | `debug` | File log level (`debug`, `info`, `warn`, `error`, `critical`) |
| `LOG_FOLDER` | (empty) | Directory for log files. If empty, file logging is disabled |
| `LOG_USE_COLOR` | `true` | Enable colored console output (`true`/`false`) |

### MPPI

| Variable | Default | Description |
|----------|---------|-------------|
| `MPPI_BACKEND` | `numpy` | Computation backend (`numpy` or `cupy` for GPU) |
| `MPPI_DTYPE` | `float64` | Floating point precision (`float16`, `float32`, or `float64`). `float16` requires `cupy` backend. |
| `MPPI_THREADS_PER_BLOCK` | `256` | CUDA threads per block (only used with `cupy` backend) |
