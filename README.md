🇫🇷 [Français](README.fr.md) | 🇬🇧 **English**

# llama.cpp Installer

Automatic installation script for llama.cpp with CUDA support for Ubuntu/Pop!_OS.

> **Compatibility**: Linux only. Tested on Pop!_OS 24.04.

## What the script does

- Updates the system
- Installs dependencies (git, cmake, build-essential, etc.)
- Optional: installs CUDA Toolkit for NVIDIA GPUs
- Clones or updates the llama.cpp repository
- Compiles llama.cpp with cmake (including `llama-server`)
- Optional: adds `llama-server` to your PATH
- Optional: downloads a sample model (Llama 2 7B)

## Usage

```bash
chmod +x install.sh
./install.sh
```

## Running a model (Qwen3.6-35B-A3B)

```bash
llama-server \
  --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
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

## Documentation

Full documentation of llama.cpp parameters and internals: [DOC.md](DOC.md)

## Testing the server

```bash
curl http://localhost:8001/health
```

## Python API example

```python
import requests

response = requests.post(
    'http://localhost:8001/chat',
    json={'messages': [{'role': 'user', 'content': 'Hello!'}]}
)
print(response.json())
```

## Author

Issa Issa