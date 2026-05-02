#!/bin/bash

# Script pour installer llama.cpp avec support pour llama-server sur Pop!_OS/Ubuntu
# Auteur : Issa Issa
# Date : 1 mai 2026

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier si le script est exécuté en tant que root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Ne pas exécuter ce script en tant que root. Utilisez un utilisateur normal.${NC}"
    exit 1
fi

# Mettre à jour le système
echo -e "${YELLOW}Mise à jour du système...${NC}"
sudo apt update && sudo apt upgrade -y

# Installer les dépendances
echo -e "${YELLOW}Installation des dépendances...${NC}"
sudo apt install -y git cmake build-essential python3 python3-pip python3-venv libblas-dev liblapack-dev libomp-dev

# Vérifier si CUDA est disponible (optionnel)
read -p "Souhaitez-vous activer le support CUDA pour GPU NVIDIA ? (o/n) : " cuda_choice
if [[ "$cuda_choice" =~ ^[Oo]$ ]]; then
    echo -e "${YELLOW}Installation de CUDA Toolkit...${NC}"
    sudo apt install -y nvidia-cuda-toolkit
    CMAKE_OPTIONS="-DGGML_CUDA=ON"
else
    CMAKE_OPTIONS=""
fi

# Cloner le dépôt llama.cpp
echo -e "${YELLOW}Clonage du dépôt llama.cpp...${NC}"
if [ -d ~/llama.cpp ]; then
    echo -e "${YELLOW}Le répertoire ~/llama.cpp existe déjà. Mise à jour et nettoyage...${NC}"
    cd ~/llama.cpp
    git reset --hard
    git clean -fd
    git pull
else
    git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
    cd ~/llama.cpp
fi

# Nettoyer le répertoire build existant
echo -e "${YELLOW}Nettoyage du répertoire build...${NC}"
rm -rf ~/llama.cpp/build

# Compiler llama.cpp avec support pour llama-server
echo -e "${YELLOW}Compilation de llama.cpp (incluant llama-server)...${NC}"
mkdir -p build && cd build
cmake $CMAKE_OPTIONS ..
cmake --build . --config Release -j$(nproc)

# Vérifier la compilation
if [ -f ./bin/llama-server ]; then
    echo -e "${GREEN}Compilation réussie ! llama-server est disponible.${NC}"
else
    echo -e "${RED}Échec de la compilation. Vérifiez les erreurs ci-dessus.${NC}"
    exit 1
fi

# Ajouter le répertoire bin au PATH (optionnel)
read -p "Souhaitez-vous ajouter ~/llama.cpp/build/bin à votre PATH pour accéder à llama-server globalement ? (o/n) : " path_choice
if [[ "$path_choice" =~ ^[Oo]$ ]]; then
    echo 'export PATH="$PATH:~/llama.cpp/build/bin"' >> ~/.bashrc
    source ~/.bashrc
    echo -e "${GREEN}Le répertoire a été ajouté à votre PATH.${NC}"
fi

# Proposer de télécharger un modèle exemple
read -p "Souhaitez-vous télécharger un modèle exemple (Llama 2 7B en GGML, ~4.5 Go) ? (o/n) : " model_choice
if [[ "$model_choice" =~ ^[Oo]$ ]]; then
    echo -e "${YELLOW}Téléchargement du modèle Llama 2 7B (GGML)...${NC}"
    wget https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/llama-2-7b.ggmlv3.q4_0.bin -O ~/llama-2-7b.ggmlv3.q4_0.bin
    if [ -f ~/llama-2-7b.ggmlv3.q4_0.bin ]; then
        echo -e "${GREEN}Téléchargement terminé : ~/llama-2-7b.ggmlv3.q4_0.bin${NC}"
    else
        echo -e "${RED}Échec du téléchargement. Vérifiez votre connexion Internet.${NC}"
    fi
fi

# Instructions finales
echo -e "${GREEN}
---
Installation terminée avec succès !

Pour exécuter un modèle avec llama-server :
1. Si vous avez ajouté le répertoire au PATH :
   Exécutez simplement :
   llama-server --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8001 --alias qwen3.6-35b-a3b -c 131072 -n 32768 --no-context-shift --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.00 --presence-penalty 0.00 --fit on -fa on -ctk q8_0 -ctv q8_0 --chat-template-kwargs '{"preserve_thinking": true}'

2. Sinon, utilisez le chemin complet :
   ~/llama.cpp/build/bin/llama-server --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8001 ...

Pour tester le serveur :
- Ouvrez un nouveau terminal et exécutez :
  curl http://localhost:8001/health

Pour utiliser l'API Python :
1. Installez le module : pip install requests
2. Exemple de code :
   import requests
   response = requests.post('http://localhost:8001/chat', json={'messages': [{'role': 'user', 'content': 'Bonjour !'}]})
   print(response.json())
${NC}"
