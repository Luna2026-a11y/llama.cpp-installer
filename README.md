🇫🇷 [Français](README.fr.md) | 🇬🇧 **English**

# llama.cpp Installer

Automatic installation script for llama.cpp with CUDA support for Ubuntu/Pop!_OS.

> **Compatibility**: Linux only. Tested on Pop!_OS 24.04 and Ubuntu 24.04.

## What the script does

- Detects NVIDIA GPU automatically
- Installs build dependencies (git, cmake, build-essential, etc.)
- Optional: installs CUDA Toolkit with NVIDIA PPA for Ubuntu
- Clones or updates the llama.cpp repository
- Compiles llama.cpp with cmake (including `llama-server` and `llama-cli`)
- Optional: adds `llama-server` to your PATH
- Optional: downloads a sample model (Qwen3-4B or Qwen3.6-35B-A3B)
- Supports `--uninstall` to cleanly remove everything

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| VRAM (NVIDIA) | 8 GB | 12 GB+ (for 35B models) |
| Disk space | 5 GB (build) + model size | 30 GB+ |
| OS | Ubuntu 22.04 / Pop!_OS 22.04 | Ubuntu 24.04 / Pop!_OS 24.04 |

**Check your GPU before starting:**
```bash
nvidia-smi  # Should show your GPU name, driver, and VRAM
```

## Quick Start

```bash
chmod +x install.sh
./install.sh
```

The script is interactive and will ask you about CUDA, PATH, and model download.

### Non-interactive Mode

```bash
# Accept all defaults (install CUDA if detected, add to PATH, download Qwen3-4B)
./install.sh -y

# Skip specific steps
./install.sh --skip-update --no-model --no-path

# Skip CUDA even on NVIDIA systems
./install.sh --no-cuda
```

### Uninstall

```bash
./install.sh --uninstall
```

## All Options

| Option | Description |
|--------|-------------|
| `--skip-update` | Skip system apt update/upgrade |
| `--no-cuda` | Skip CUDA installation even if NVIDIA GPU detected |
| `--no-model` | Skip sample model download |
| `--no-path` | Skip adding llama-server to PATH |
| `--uninstall` | Remove llama.cpp installation |
| `-y` / `--yes` | Accept all defaults (non-interactive) |
| `-h` / `--help` | Show help |

## Running a Model

### Small model — Qwen3-4B (8GB+ VRAM)

```bash
llama-server \
  --model ~/models/Qwen3-4B-Q4_K_M.gguf \
  --port 8001 \
  --alias qwen3-4b \
  -ngl 999 \
  -c 8192 \
  -fa on
```

### Large MoE model — Qwen3.6-35B-A3B (12GB+ VRAM)

```bash
llama-server \
  --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --port 8001 \
  --alias qwen3.6-35b-a3b \
  -ngl 999 \
  -ncmoe 0 \
  -c 131072 \
  -n 32768 \
  --no-context-shift \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --repeat-penalty 1.00 \
  --presence-penalty 0.00 \
  --fit on \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --chat-template-kwargs '{"preserve_thinking": true}'
```

> **Desktop users**: A running X server/compositor uses ~500 MiB of VRAM. Reduce `-c` by ~30-40k tokens on desktop vs. headless. See [DOC.md](DOC.md) for details.

## Downloading Models

Models are in **GGUF** format (the current standard). Download from HuggingFace:

| Model | Quant | Size | VRAM needed | Link |
|-------|-------|------|-------------|------|
| Qwen3-4B | Q4_K_M | ~2.5 GB | 8 GB+ | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Qwen3.6-35B-A3B | UD-Q4_K_M | ~18 GB | 12 GB+ | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.6-27B | IQ4_XS | ~14.7 GB | 16 GB+ | [Various](https://huggingface.co/models?search=qwen3.6-27b+gguf) |

```bash
# Example: download Qwen3-4B
mkdir -p ~/models
curl -L -o ~/models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

> **Important**: Only download `.gguf` files. The older `.ggml` / `.bin` format is no longer supported.

## Running as a Systemd Service

Create `/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=llama.cpp Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/llama.cpp/build/bin/llama-server \
  --model /home/YOUR_USERNAME/models/Qwen3-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8001 \
  -ngl 999 -fa on -c 8192
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Check status
sudo systemctl status llama-server

# View logs
journalctl -u llama-server -f
```

## Testing the Server

```bash
# Health check
curl http://localhost:8001/health

# Chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```

## Python API Example

```python
import requests

response = requests.post(
    'http://localhost:8001/v1/chat/completions',
    json={
        'model': 'default',
        'messages': [{'role': 'user', 'content': 'Hello!'}]
    }
)
print(response.json())
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA drivers: `sudo apt install nvidia-driver-550` |
| CUDA OOM | Reduce `-c`, increase `-ncmoe`, or switch KV cache to `q4_0` |
| Slow inference | Verify `-fa on` and `-ngl 999` are set |
| `llama-server: command not found` | Run `source ~/.bashrc` or use full path |
| Build fails on cmake | Install build deps: `sudo apt install cmake build-essential` |

See [DOC.md](DOC.md) for the complete parameter reference.

## Author

Issa Issa