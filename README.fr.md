🇫🇷 **Français** | 🇬🇧 [English](README.md)

# llama.cpp Installer

Script d'installation automatique de llama.cpp avec support CUDA pour Ubuntu/Pop!_OS.

> **Compatibilité** : Linux uniquement. Testé sous Pop!_OS 24.04.

## Ce que fait le script

- Met à jour le système
- Installe les dépendances (git, cmake, build-essential, etc.)
- Optionnel : installe le CUDA Toolkit pour les GPU NVIDIA
- Clone ou met à jour le dépôt llama.cpp
- Compile llama.cpp avec cmake (incluant `llama-server`)
- Optionnel : ajoute `llama-server` au PATH
- Optionnel : télécharge un modèle exemple (Llama 2 7B)

## Utilisation

```bash
chmod +x install.sh
./install.sh
```

## Exécuter un modèle (Qwen3.6-35B-A3B)

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

Documentation complète des paramètres llama.cpp et fonctionnement interne : [DOC.fr.md](DOC.fr.md)

## Tester le serveur

```bash
curl http://localhost:8001/health
```

## Exemple API Python

```python
import requests

response = requests.post(
    'http://localhost:8001/chat',
    json={'messages': [{'role': 'user', 'content': 'Bonjour !'}]}
)
print(response.json())
```

## Auteur

Issa Issa