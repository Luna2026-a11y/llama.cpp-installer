🇫🇷 **Français** | 🇬🇧 [English](README.md)

# llama.cpp Installer

Script d'installation automatique de llama.cpp avec support CUDA pour Ubuntu/Pop!_OS.

> **Compatibilité** : Linux uniquement. Testé sur Pop!_OS 24.04 et Ubuntu 24.04.

## Ce que fait le script

- Détecte automatiquement la carte NVIDIA
- Installe les dépendances de compilation (git, cmake, build-essential, etc.)
- Optionnel : installe le CUDA Toolkit avec le PPA NVIDIA sur Ubuntu
- Clone ou met à jour le dépôt llama.cpp
- Compile llama.cpp avec cmake (y compris `llama-server` et `llama-cli`)
- Optionnel : ajoute `llama-server` au PATH
- Optionnel : télécharge un modèle exemple (Qwen3-4B ou Qwen3.6-35B-A3B)
- Supporte `--uninstall` pour tout supprimer proprement

## Prérequis

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| RAM | 8 Go | 16 Go+ |
| VRAM (NVIDIA) | 8 Go | 12 Go+ (pour les modèles 35B) |
| Espace disque | 5 Go (build) + taille du modèle | 30 Go+ |
| OS | Ubuntu 22.04 / Pop!_OS 22.04 | Ubuntu 24.04 / Pop!_OS 24.04 |

**Vérifiez votre GPU avant de commencer :**
```bash
nvidia-smi  # Doit afficher le nom du GPU, le driver et la VRAM
```

## Démarrage rapide

```bash
chmod +x install.sh
./install.sh
```

Le script est interactif et vous demandera pour CUDA, le PATH et le téléchargement de modèle.

### Mode non-interactif

```bash
# Accepter tous les defaults (installer CUDA si détecté, ajouter au PATH, télécharger Qwen3-4B)
./install.sh -y

# Ignorer certaines étapes
./install.sh --skip-update --no-model --no-path

# Ignorer CUDA même sur un système NVIDIA
./install.sh --no-cuda
```

### Désinstallation

```bash
./install.sh --uninstall
```

## Toutes les options

| Option | Description |
|--------|-------------|
| `--skip-update` | Ignorer la mise à jour apt |
| `--no-cuda` | Ignorer CUDA même si un GPU NVIDIA est détecté |
| `--no-model` | Ignorer le téléchargement du modèle |
| `--no-path` | Ignorer l'ajout au PATH |
| `--uninstall` | Supprimer l'installation llama.cpp |
| `-y` / `--yes` | Accepter tous les defaults (non-interactif) |
| `-h` / `--help` | Afficher l'aide |

## Exécuter un modèle

### Petit modèle — Qwen3-4B (8 Go+ VRAM)

```bash
llama-server \
  --model ~/models/Qwen3-4B-Q4_K_M.gguf \
  --port 8001 \
  --alias qwen3-4b \
  -ngl 999 \
  -c 8192 \
  -fa on
```

### Grand modèle MoE — Qwen3.6-35B-A3B (12 Go+ VRAM)

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

> **Utilisateurs desktop** : Un serveur X/compositeur utilise ~500 MiO de VRAM. Réduisez `-c` de ~30-40k tokens par rapport au mode headless. Voir [DOC.fr.md](DOC.fr.md) pour les détails.

## Télécharger des modèles

Les modèles sont au format **GGUF** (le standard actuel). Téléchargez-les depuis HuggingFace :

| Modèle | Quant | Taille | VRAM nécessaire | Lien |
|--------|-------|--------|-----------------|------|
| Qwen3-4B | Q4_K_M | ~2,5 Go | 8 Go+ | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Qwen3.6-35B-A3B | UD-Q4_K_M | ~18 Go | 12 Go+ | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.6-27B | IQ4_XS | ~14,7 Go | 16 Go+ | [Divers](https://huggingface.co/models?search=qwen3.6-27b+gguf) |

```bash
# Exemple : télécharger Qwen3-4B
mkdir -p ~/models
curl -L -o ~/models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

> **Important** : Ne téléchargez que les fichiers `.gguf`. L'ancien format `.ggml` / `.bin` n'est plus supporté.

## Lancer en tant que service systemd

Créer `/etc/systemd/system/llama-server.service` :

```ini
[Unit]
Description=llama.cpp Server
After=network.target

[Service]
Type=simple
User=VOTRE_UTILISATEUR
ExecStart=/home/VOTRE_UTILISATEUR/llama.cpp/build/bin/llama-server \
  --model /home/VOTRE_UTILISATEUR/models/Qwen3-4B-Q4_K_M.gguf \
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

# Vérifier le statut
sudo systemctl status llama-server

# Voir les logs
journalctl -u llama-server -f
```

## Tester le serveur

```bash
# Vérification de santé
curl http://localhost:8001/health

# Chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Bonjour !"}]}'
```

## Exemple API Python

```python
import requests

response = requests.post(
    'http://localhost:8001/v1/chat/completions',
    json={
        'model': 'default',
        'messages': [{'role': 'user', 'content': 'Bonjour !'}]
    }
)
print(response.json())
```

## Dépannage

| Problème | Solution |
|----------|----------|
| `nvidia-smi` introuvable | Installer les drivers NVIDIA : `sudo apt install nvidia-driver-550` |
| OOM CUDA | Réduire `-c`, augmenter `-ncmoe`, ou passer le cache KV en `q4_0` |
| Inférence lente | Vérifier `-fa on` et `-ngl 999` |
| `llama-server: command not found` | Lancer `source ~/.bashrc` ou utiliser le chemin complet |
| Échec de compilation cmake | Installer les dépendances : `sudo apt install cmake build-essential` |

Voir [DOC.fr.md](DOC.fr.md) pour la référence complète des paramètres.

## Pour aller plus loin

Cet installateur permet de faire tourner llama.cpp sur votre machine. Pour du **serving de niveau production** — configs multi-moteur (vLLM, SGLang), recettes Docker Compose benchmarquées, patches Genesis, et tuning par workload — consultez [club-3090](https://github.com/noonghunna/club-3090). C'est la référence pour faire tourner des LLMs sur des GPUs grand public avec des TPS mesurés, des budgets VRAM, et des configs testées en conditions réelles.

## Auteur

Issa Issa