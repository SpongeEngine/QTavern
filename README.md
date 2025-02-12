# SpongeQuant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)

Open-source web UI for quantizing LLMs from Hugging Face. Inspired by [Oobabooga](https://github.com/oobabooga/text-generation-webui) and the [Colab AutoQuant notebook](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4).  

## Quick Start

### Windows
#### Install Docker Desktop
Got to https://docs.docker.com/desktop/setup/install/windows-install/, download and install Docker Desktop.
```
Right click on `start_windows.ps1` and press `Run with PowerShell`.
```

### Linux
#### Install Docker
```bash
curl -fsSL https://get.docker.com | sh
```
#### Run the script
```bash
./start_linux.sh
```

## Features
- **Multi-method Quantization:** Choose from GGUF, GPTQ, ExLlamaV2, AWQ, and HQQ.
- **Unified Docker Setup:** Separate Dockerfiles for CPU-only and GPU (CUDA) builds.
- **Dynamic Runtime Detection:** Launch the appropriate container based on hardware (via startup scripts for Linux and Windows).
- **Easy-to-Use Web UI:** Built with Gradio, enabling interactive model quantization.

# Quantization Methods Comparison

| Method       | CPU Quantization       | CPU Inference         | GPU Quantization       | GPU Inference          | Tradeoffs / Notes                                                                                           |
|--------------|------------------------|-----------------------|------------------------|------------------------|-------------------------------------------------------------------------------------------------------------|
| **GGUF**     | Yes                    | Yes                   | Not Required           | Not Required           | Designed for efficient CPU inference via llama.cpp; optimized for low precision on CPUs.                      |
| **GPTQ**     | Poor/Not Recommended   | Poor                  | Yes                    | Yes                    | High compression & accuracy but built for CUDA; forcing CPU-only leads to very slow and unreliable processing.|
| **ExLlamaV2**| Limited (Experimental) | Limited               | Yes                    | Yes                    | Optimized for GPU; CPU fallback is possible but performance is suboptimal.                                  |
| **AWQ**      | Limited                | Poor/Not Recommended   | Yes                    | Yes                    | Relies on CUDA kernels for fast quantization; CPU-only execution is generally impractical.                    |
| **HQQ**      | Limited                | Limited/Unvalidated   | Yes                    | Yes                    | Designed primarily for GPU inference with specialized kernels; CPU usage is not widely validated and may be very slow. |
- **GGUF** is the recommended method for CPU-only quantization due to its design for CPU-friendly inference.
- **GPTQ**, **ExLlamaV2**, **AWQ**, and **HQQ** are optimized for GPU-based quantization. They can sometimes be forced to run on CPU by using dummy CUDA paths or environment variables, but performance and reliability may suffer significantly.
- Future improvements or updates in these methods might expand their CPU support, but as of now, only GGUF is reliably CPU-friendly.

### Project Structure
```
SpongeQuant/
├── app/
│   ├── app.py                # Main application code (Gradio UI)
│   ├── requirements.cpu.txt  # CPU-only dependencies
│   ├── requirements.gpu-cuda.txt  # GPU (CUDA) dependencies
│   └── ...                   # Other application files
├── Dockerfile.cpu            # Dockerfile for CPU-only mode
├── Dockerfile.gpu-cuda       # Dockerfile for GPU (CUDA) mode
├── Dockerfile.gpu-rocm       # (Placeholder for future ROCm support)
├── start_linux.sh            # Startup script for Linux
├── start_windows.ps1         # Startup script for Windows
├── README.md                 # This file
└── ...                       # Other files (models, quantized_models, etc.)
```

### Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests on GitHub.


```
docker run --gpus all -it -p "${PORT}:${PORT}" \
      -v "$(pwd)/app/gguf:/app/gguf" \
      -v "$(pwd)/models:/app/models" \
      -v "$(pwd)/quantized_models:/app/quantized_models" \
      --rm "${IMAGE_NAME}"
```

```
docker run -it -p "${PORT}:${PORT}" \
      -v "$(pwd)/app/gguf:/app/gguf" \
      -v "$(pwd)/models:/app/models" \
      -v "$(pwd)/quantized_models:/app/quantized_models" \
      --rm "${IMAGE_NAME}"
```

x86-64 CPUs have AVX2/FMA support, which can accelerate tensor operations in llama.cpp much faster than ARM NEON/DOTPROD.
