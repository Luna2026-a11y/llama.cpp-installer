#!/bin/bash
#
# llama.cpp Installer — Ubuntu/Pop!_OS with CUDA support
# Author: Issa Issa
# License: MIT
#
# Usage: chmod +x install.sh && ./install.sh
# Options:
#   --skip-update    Skip system apt update/upgrade
#   --no-cuda        Skip CUDA installation even if NVIDIA GPU detected
#   --no-model       Skip sample model download
#   --no-path        Skip adding llama-server to PATH
#   --uninstall      Remove llama.cpp installation
#   -y / --yes       Accept all defaults (non-interactive)
#

set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Defaults ─────────────────────────────────────────────────────────────────
SKIP_UPDATE=false
NO_CUDA=false
NO_MODEL=false
NO_PATH=false
UNINSTALL=false
NON_INTERACTIVE=false
LLAMA_DIR="$HOME/llama.cpp"
MODEL_DIR="$HOME/models"

# ─── Parse arguments ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-update)   SKIP_UPDATE=true ;;
        --no-cuda)       NO_CUDA=true ;;
        --no-model)      NO_MODEL=true ;;
        --no-path)       NO_PATH=true ;;
        --uninstall)     UNINSTALL=true ;;
        -y|--yes)        NON_INTERACTIVE=true ;;
        --help|-h)
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --skip-update   Skip apt update/upgrade"
            echo "  --no-cuda       Skip CUDA even if NVIDIA GPU detected"
            echo "  --no-model      Skip sample model download"
            echo "  --no-path       Skip adding to PATH"
            echo "  --uninstall     Remove llama.cpp installation"
            echo "  -y, --yes       Accept all defaults (non-interactive)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}" >&2
            echo "Run ./install.sh --help for usage." >&2
            exit 1
            ;;
    esac
done

# ─── Helper functions ─────────────────────────────────────────────────────────

info()  { echo -e "${CYAN}ℹ${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1" >&2; exit 1; }

ask_yes() {
    local prompt="$1"
    local default="${2:-n}"
    if [[ "$NON_INTERACTIVE" == true ]]; then
        [[ "$default" =~ ^[OoYy]$ ]] && return 0 || return 1
    fi
    local options="O/n"
    [[ "$default" =~ ^[OoYy]$ ]] && options="O/n" || options="o/N"
    read -p "$(echo -e "${YELLOW}${prompt} (${options}) : ${NC}")" answer
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[OoYy]$ ]]
}

cmd_exists()  { command -v "$1" &>/dev/null; }

# ─── Uninstall ────────────────────────────────────────────────────────────────
if [[ "$UNINSTALL" == true ]]; then
    echo -e "${BOLD}Uninstalling llama.cpp...${NC}"
    if [[ -d "$LLAMA_DIR" ]]; then
        rm -rf "$LLAMA_DIR"
        ok "Removed $LLAMA_DIR"
    else
        warn "$LLAMA_DIR not found"
    fi
    # Remove from PATH in .bashrc
    if grep -q 'llama.cpp/build/bin' ~/.bashrc 2>/dev/null; then
        sed -i '/llama\.cpp\/build\/bin/d' ~/.bashrc
        ok "Removed llama.cpp from PATH in .bashrc"
    fi
    # Optionally remove model dir
    if ask_yes "Also remove $MODEL_DIR?"; then
        rm -rf "$MODEL_DIR"
        ok "Removed $MODEL_DIR"
    fi
    ok "Uninstall complete. Run 'source ~/.bashrc' or open a new terminal."
    exit 0
fi

# ─── Pre-flight checks ────────────────────────────────────────────────────────

echo -e "${BOLD}━━━ llama.cpp Installer ━━━${NC}"
echo ""

# Must NOT be root
if [[ "$EUID" -eq 0 ]]; then
    fail "Do not run this script as root. Use a normal user account."
fi

# Detect OS
if [[ ! -f /etc/os-release ]]; then
    fail "Cannot detect OS. This script supports Ubuntu/Pop!_OS only."
fi
source /etc/os-release
case "$ID" in
    ubuntu|pop) info "Detected OS: $NAME $VERSION" ;;
    linuxmint) info "Detected OS: $NAME $VERSION (Ubuntu-based)" ;;
    debian)    warn "Debian detected — works but not tested. Proceeding anyway." ;;
    *)         fail "Unsupported OS: $ID. This script requires Ubuntu/Pop!_OS or derivatives." ;;
esac

# ─── Step 1: System update ────────────────────────────────────────────────────
if [[ "$SKIP_UPDATE" == false ]]; then
    info "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    ok "System updated"
else
    info "Skipping system update (--skip-update)"
fi

# ─── Step 2: Install dependencies ────────────────────────────────────────────
info "Installing build dependencies..."
sudo apt install -y git cmake build-essential python3 python3-pip python3-venv \
    libblas-dev liblapack-dev libomp-dev curl
ok "Dependencies installed"

# ─── Step 3: Detect NVIDIA GPU ────────────────────────────────────────────────
HAS_NVIDIA=false
CMAKE_OPTIONS=""

if [[ "$NO_CUDA" == true ]]; then
    info "CUDA skipped (--no-cuda)"
else
    if lspci 2>/dev/null | grep -qi 'nvidia'; then
        HAS_NVIDIA=true
        info "NVIDIA GPU detected: $(lspci 2>/dev/null | grep -i 'nvidia' | head -1 | sed 's/.*: //')"

        # Check nvidia driver
        if cmd_exists nvidia-smi; then
            DRV_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
            ok "GPU: $GPU_NAME, Driver: $DRV_VER, VRAM: ${VRAM_MB} MiB"
        else
            warn "nvidia-smi not found. Install NVIDIA drivers first."
        fi

        # Install CUDA
        if ask_yes "Install CUDA Toolkit for NVIDIA GPU acceleration?" "o"; then
            info "Installing CUDA Toolkit..."
            # Pop!_OS has CUDA in system76 repos; Ubuntu needs the NVIDIA PPA
            if [[ "$ID" == "pop" ]]; then
                sudo apt install -y nvidia-cuda-toolkit
            else
                # Ubuntu: use the NVIDIA CUDA PPA for newer versions
                if ! dpkg -l | grep -q 'cuda-toolkit' 2>/dev/null; then
                    info "Adding NVIDIA CUDA repository..."
                    sudo apt install -y software-properties-common
                    sudo add-apt-repository -y "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /"
                    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/319fa0b6-13b0.pub | sudo apt-key add - 2>/dev/null || true
                    sudo apt update
                    sudo apt install -y nvidia-cuda-toolkit
                fi
            fi
            CMAKE_OPTIONS="-DGGML_CUDA=ON"
            ok "CUDA Toolkit installed"
        fi
    else
        info "No NVIDIA GPU detected — will compile CPU-only"
    fi
fi

# ─── Step 4: Clone or update llama.cpp ────────────────────────────────────────
if [[ -d "$LLAMA_DIR" ]]; then
    info "Updating existing llama.cpp at $LLAMA_DIR ..."
    cd "$LLAMA_DIR"

    # Check for local changes before resetting
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        warn "Local changes detected in $LLAMA_DIR"
        if ask_yes "Discard local changes and pull latest?"; then
            git stash 2>/dev/null || true
        else
            info "Keeping local changes. Pulling with rebase..."
            git pull --rebase || warn "Pull failed. Building from current state."
        fi
    else
        git pull || warn "Pull failed. Building from current state."
    fi
else
    info "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
fi

# ─── Step 5: Compile ──────────────────────────────────────────────────────────
info "Compiling llama.cpp (this may take 5-10 minutes)..."
rm -rf build
mkdir -p build && cd build

if ! cmake $CMAKE_OPTIONS .. 2>&1 | tail -5; then
    fail "CMake configuration failed. Check error messages above."
fi

if ! cmake --build . --config Release -j"$(nproc)" 2>&1 | tail -5; then
    fail "Compilation failed. Check error messages above."
fi

# Verify binaries
BINARY_PATH=""
if [[ -f ./bin/llama-server ]]; then
    BINARY_PATH="$LLAMA_DIR/build/bin"
    ok "llama-server compiled successfully"
else
    # Some builds put binaries in different locations
    FOUND=$(find . -name "llama-server" -type f 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        BINARY_PATH="$LLAMA_DIR/build/$(dirname "$FOUND")"
        ok "llama-server compiled successfully at $FOUND"
    else
        fail "llama-server binary not found after compilation."
    fi
fi

# Also check for llama-cli
if [[ -f ./bin/llama-cli ]] || find . -name "llama-cli" -type f 2>/dev/null | grep -q .; then
    ok "llama-cli also available"
fi

# ─── Step 6: Add to PATH ──────────────────────────────────────────────────────
if [[ "$NO_PATH" == false ]]; then
    if ask_yes "Add llama-server to your PATH? (so you can run it from anywhere)" "o"; then
        # Remove old entries first
        sed -i '/llama\.cpp\/build\/bin/d' ~/.bashrc 2>/dev/null || true
        echo "export PATH=\"\$PATH:$BINARY_PATH\"" >> ~/.bashrc
        export PATH="$PATH:$BINARY_PATH"
        ok "Added to PATH. Run 'source ~/.bashrc' or open a new terminal."
    fi
else
    info "Binary location: $BINARY_PATH/llama-server"
fi

# ─── Step 7: Download sample model ────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

if [[ "$NO_MODEL" == false ]]; then
    echo ""
    echo -e "${BOLD}━━━ Sample Models ━━━${NC}"
    echo "  1) Qwen3-4B (Q4_K_M)         — ~2.5 GB — Small, fast, good for testing"
    echo "  2) Qwen3.6-35B-A3B (Q4_K_M)  — ~18 GB — MoE, powerful, needs 12GB+ VRAM"
    echo "  3) Skip model download"
    echo ""

    MODEL_CHOICE="1"
    if [[ "$NON_INTERACTIVE" == false ]]; then
        read -p "$(echo -e "${YELLOW}Which model to download? [1/2/3] (default: 1): ${NC}")" MODEL_CHOICE
        MODEL_CHOICE="${MODEL_CHOICE:-1}"
    fi

    case "$MODEL_CHOICE" in
        1)
            MODEL_URL="https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
            MODEL_FILE="$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf"
            MODEL_SIZE="~2.5 GB"
            ;;
        2)
            MODEL_URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            MODEL_FILE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            MODEL_SIZE="~18 GB"
            ;;
        3)
            info "Skipping model download"
            MODEL_URL=""
            ;;
        *)
            warn "Invalid choice. Skipping model download"
            MODEL_URL=""
            ;;
    esac

    if [[ -n "$MODEL_URL" ]]; then
        if [[ -f "$MODEL_FILE" ]]; then
            ok "Model already exists: $MODEL_FILE"
        else
            info "Downloading model ($MODEL_SIZE)..."
            echo "  URL: $MODEL_URL"
            if curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"; then
                ok "Model downloaded: $MODEL_FILE"
            else
                rm -f "$MODEL_FILE" 2>/dev/null
                fail "Model download failed. You can download manually from HuggingFace."
            fi
        fi
    fi
fi

# ─── Step 8: Generate config snippets ──────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Installation Complete! ━━━${NC}"
echo ""

# Quick start command
if [[ -n "$BINARY_PATH" ]]; then
    echo -e "${CYAN}Quick start:${NC}"
    if [[ -f "$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf" ]]; then
        echo "  llama-server --model ~/models/Qwen3-4B-Q4_K_M.gguf --port 8001 -ngl 999 -c 8192 -fa on"
    elif [[ -f "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" ]]; then
        echo "  llama-server --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8001 -ngl 999 -ncmoe 0 -c 131072 -n 32768 --no-context-shift --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.00 --presence-penalty 0.00 --fit on -fa on -ctk q8_0 -ctv q8_0 --chat-template-kwargs '{\"preserve_thinking\": true}'"
    else
        echo "  llama-server --model <path-to-model.gguf> --port 8001 -ngl 999 -c 8192 -fa on"
    fi
    echo ""
    echo -e "${CYAN}Test the server:${NC}"
    echo "  curl http://localhost:8001/health"
    echo ""
    echo -e "${CYAN}Python API example:${NC}"
    echo '  python3 -c "import requests; r=requests.post('"'"'http://localhost:8001/v1/chat/completions'"'"', json={'"'"'model'"'"':'"'"'default'"'"','"'"'messages'"'"':[{'"'"'role'"'"':'"'"'user'"'"','"'"'content'"'"':'"'"'Hello!'"'"'}]}); print(r.json())"'
fi

echo ""
echo -e "${CYAN}See DOC.md for full parameter reference.${NC}"
echo -e "${CYAN}See README.md for systemd service setup and advanced usage.${NC}"

# ─── Reminder about PATH ─────────────────────────────────────────────────────
if ! echo "$PATH" | grep -q 'llama.cpp/build/bin'; then
    echo ""
    warn "llama-server is not yet in your current shell PATH."
    info "Run: source ~/.bashrc"
    info "Or use the full path: $BINARY_PATH/llama-server"
fi