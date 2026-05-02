# Documentation llama.cpp — Paramètres et Fonctionnement

> Référence complète des options de ligne de commande de `llama-server` et du moteur `llama.cpp`.

---

## Sommaire

1. [Architecture générale](#architecture-générale)
2. [Paramètres de modèle](#paramètres-de-modèle)
3. [Paramètres de contexte](#paramètres-de-contexte)
4. [Paramètres GPU / CUDA](#paramètres-gpu--cuda)
5. [Paramètres MoE (Mixture of Experts)](#paramètres-moe-mixture-of-experts)
6. [Paramètres de cache KV](#paramètres-de-cache-kv)
7. [Paramètres de sampling / génération](#paramètres-de-sampling--génération)
8. [Paramètres serveur](#paramètres-serveur)
9. [Paramètres de performance](#paramètres-de-performance)
10. [Fonctionnement interne](#fonctionnement-interne)
11. [Cas pratique : Qwen3.6-35B-A3B](#cas-pratique--qwen36-35b-a3b)
12. [Dépannage](#dépannage)

---

## Architecture générale

### Comment fonctionne llama.cpp

`llama.cpp` est un moteur d'inférence LLM écrit en C/C++, optimisé pour tourner sur du matériel grand public. Il ne fait pas d'entraînement — il ne fait que de l'inférence (génération de texte).

Le flux de données :

```
Fichier GGUF → Chargement en mémoire → Déquantification par couche → Inférence → Tokens générés
```

### Deux exécutables principaux

| Binaire | Rôle |
|---------|------|
| `llama-server` | Serveur HTTP avec API compatible OpenAI (`/v1/chat/completions`) |
| `llama-cli` | Interface CLI interactive pour le chat en terminal |

### Le format GGUF

GGUF (GPT-Generated Unified Format) est le format de modèle de llama.cpp. Il contient :
- Les poids du modèle (quantifiés en Q4, Q5, Q8, etc.)
- Les métadonnées (tokenizer, hyperparamètres, template de chat)
- Les tensors organisés par couche

Les niveaux de quantification courants :

| Quant | Taille approx. | Qualité | Usage recommandé |
|-------|---------------|---------|------------------|
| `Q3_K_M` | Plus petit | Bonne | RAM très limitée |
| `Q4_K_M` | ~40% du FP16 | Très bon | Sweet spot pour les GPU 12GB |
| `Q4_K_XL` | Un peu plus gros | Meilleur | GPU avec plus de VRAM |
| `Q5_K_M` | ~55% du FP16 | Excellent | Quand la VRAM le permet |
| `Q6_K` | ~65% du FP16 | Quasi-original | Presque sans perte |
| `Q8_0` | ~75% du FP16 | Quasi-parfait | Quantification quasi-transparente |
| `UD-Q4_K_XL` | Variable | Unsloth Dynamic | Quantification adaptative (meilleur rapport poids/qualité) |

> **Note** : Les quantifications `UD-*` (Unsloth Dynamic) sont des quantifications adaptatives qui ajustent le niveau de précision par tensor, offrant un meilleur rapport qualité/taille que les quantifications uniformes.

---

## Paramètres de modèle

### `-m, --model <chemin>` — Chemin vers le fichier GGUF

Le chemin vers le fichier modèle quantifié.

```bash
-m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### `--alias <nom>` — Alias pour le modèle

Nom d'affichage utilisé dans l'API et les logs. Pratique quand on lance plusieurs modèles.

```bash
--alias qwen35b
```

Dans l'API, le modèle sera référencé comme `qwen35b` au lieu du chemin complet.

---

## Paramètres de contexte

### `-c, --ctx-size <n>` — Taille du contexte (tokens)

Le nombre maximum de tokens que le modèle peut "voir" en mémoire. C'est la fenêtre de contexte.

```bash
-c 131072   # 128k tokens
-c 32768    # 32k tokens
-c 8192     # 8k tokens (défaut)
```

**Impact** :
- Plus de contexte = plus de VRAM/RAM consommée par le cache KV
- 128k context = ~8 GB de cache KV en Q8_0 pour un 35B MoE
- La vitesse de préremplissage (prompt processing) diminue avec la taille du contexte

### `-n, --predict <n>` — Nombre maximum de tokens à générer

Limite haute pour la génération. Le modèle peut s'arrêter avant s'il rencontre un token de fin.

```bash
-n 32768    # Générer jusqu'à 32k tokens
```

### `--no-context-shift` — Désactiver le glissement de contexte

Par défaut, quand le contexte est plein, llama.cpp peut glisser la fenêtre (retirer les tokens les plus anciens) pour continuer la conversation. Ce flag désactive ce comportement.

**Quand l'utiliser** : Quand vous voulez que le modèle respecte strictement la taille de contexte définie, sans tronquer silencieusement l'historique.

---

## Paramètres GPU / CUDA

### `-ngl, --n-gpu-layers <n>` — Nombre de couches sur le GPU

Définit combien de couches du modèle sont déplacées vers la VRAM du GPU. `999` = toutes les couches.

```bash
-ngl 999    # Tout sur le GPU (recommandé si VRAM suffisante)
-ngl 0      # Tout sur le CPU (très lent)
-ngl 20     # Seulement 20 couches sur GPU (Partial offloading)
```

**Impact** :
- Plus de couches sur GPU = plus rapide, mais plus de VRAM
- Si tout tient en VRAM, utilisez `-ngl 999`
- Si VRAM insuffisante, réduisez progressivement jusqu'à ce que ça tienne

### `-sm, --split-mode <mode>` — Mode de répartition multi-GPU

| Mode | Description |
|------|-------------|
| `row` | Répartition par lignes (défaut) — équilibre entre les GPUs |
| `layer` | Répartition par couches — chaque GPU gère des couches entières |

Utile uniquement avec plusieurs GPUs NVIDIA.

---

## Paramètres MoE (Mixture of Experts)

### `-ncmoe, --n-cmoe-offload <n>` — Nombre d'experts MoE sur le GPU

**C'est le paramètre clé pour les modèles MoE sur GPU limité.**

Les modèles MoE (comme Qwen3.6-35B-A3B, Mixtral, DeepSeek) ont plusieurs "experts" par couche, mais n'en activent que quelques-uns par token. Le défi : tous les experts doivent être en mémoire pour choisir lesquels activer.

Le flag `-ncmoe` contrôle combien d'experts sont chargés sur le GPU :

```bash
-ncmoe 25   # 25 experts sur GPU — sweet spot pour 12GB VRAM
-ncmoe 0    # Aucun expert MoE sur GPU (tous en RAM)
```

**Comment ça marche** :

1. Le modèle MoE a N experts par couche (e.g., 64 experts pour Qwen3.6)
2. À chaque token, seuls quelques experts sont activés (e.g., 8 sur 64)
3. Mais TOUS les experts doivent être accessibles pour la sélection
4. `-ncmoe 25` garde les 25 premiers experts en VRAM, le reste en RAM système
5. Les experts les plus fréquemment utilisés sont privilégiés sur le GPU

**Impact sur les performances** :

| Valeur | VRAM | Vitesse | Recommandation |
|--------|------|---------|----------------|
| `-ncmoe 0` | ~4 GB | 25-30 tok/s | GPU très limité |
| `-ncmoe 25` | ~6.5-10.6 GB | 58-62 tok/s | **RTX 4070 12GB** ★ |
| `-ncmoe 50` | ~14 GB | 60-65 tok/s | RTX 4070 Ti 16GB |
| Tous sur GPU | ~20+ GB | 65+ tok/s | RTX 4090 24GB |

---

## Paramètres de cache KV

### `--cache-type-k <type>` — Type de quantification du cache KV (clés)

### `--cache-type-v <type>` — Type de quantification du cache KV (valeurs)

Le cache KV (Key-Value) stocke les représentations intermédiaires des tokens déjà traités, pour ne pas les recalculer à chaque nouveau token.

```bash
--cache-type-k q8_0   # Clés en Q8 (8-bit quantifié)
--cache-type-v q8_0   # Valeurs en Q8 (8-bit quantifié)
```

**Types disponibles** :

| Type | Précision | VRAM | Cohérence long contexte |
|------|-----------|------|--------------------------|
| `f16` | Complète | 2x | Référence |
| `q8_0` | 8-bit | 1x | Très bonne ★ |
| `q4_0` | 4-bit | 0.5x | Dégradation notable |
| `q4_1` | 4-bit amélioré | 0.5x | Acceptable |

**Pourquoi Q8_0 est le sweet spot** :
- Divise la VRAM du cache par 2 par rapport au FP16
- Préserve suffisamment de précision pour les longs contextes (128k)
- La dégradation est imperceptible en pratique
- Q4_0 est trop agressif — le modèle perd en cohérence au-delà de 32k tokens

**Impact sur 128k de contexte** :
- FP16 : ~16 GB de cache KV → impossible en 12 GB VRAM
- Q8_0 : ~8 GB de cache KV → tient dans 12 GB VRAM avec de la marge
- Q4_0 : ~4 GB mais qualité dégradée

---

## Paramètres de sampling / génération

### `--temp <valeur>` — Température

Contrôle la "créativité" du modèle. Plus c'est bas, plus c'est déterministe.

```bash
--temp 0.6   # Bon pour le coding — créatif mais cohérent
--temp 0.0   # Greedy decoding — toujours le même output
--temp 1.0   # Très aléatoire
```

**Valeurs recommandées** :
- Coding/analyse : 0.3 - 0.6
- Chat créatif : 0.7 - 0.9
- Tâches déterministes : 0.0

### `--top-p <valeur>` — Nucleus sampling

Ne considère que les tokens dont la probabilité cumulée atteint ce seuil.

```bash
--top-p 0.95   # Les tokens couvrant 95% de la probabilité
```

### `--top-k <valeur>` — Top-K sampling

Ne considère que les K tokens les plus probables.

```bash
--top-k 20   # Seulement les 20 tokens les plus probables
```

### `--repeat-penalty <valeur>` — Pénalité de répétition

Augmente la pénalité pour les tokens déjà générés, évitant les boucles.

```bash
--repeat-penalty 1.00   # Pas de pénalité (recommandé pour MoE avec thinking)
```

> **Note** : Pour les modèles avec "thinking" (comme Qwen3), une pénalité de répétition à 1.0 est recommandée car le modèle gère naturellement la structure de sa pensée.

### `--presence-penalty <valeur>` — Pénalité de présence

Pénalise les tokens qui ont déjà apparu, encourageant la diversité.

```bash
--presence-penalty 0.00   # Pas de pénalité supplémentaire
```

### `--chat-template-kwargs` — Arguments du template de chat

Paramètres supplémentaires passés au template de chat du modèle.

```bash
--chat-template-kwargs '{"preserve_thinking": true}'
```

`preserve_thinking: true` est **essentiel** pour les modèles avec mode réflexion (comme Qwen3). Sans ça, le contenu de réflexion (`<think>...</think>`) est silencieusement supprimé dans les réponses API.

---

## Paramètres serveur

### `--host <adresse>` — Adresse d'écoute

```bash
--host 0.0.0.0   # Écoute sur toutes les interfaces (accès réseau)
--host 127.0.0.1 # Écoute en local uniquement
```

### `--port <port>` — Port d'écoute

```bash
--port 8081    # Port personnalisé
--port 8001    # Autre port
```

### `--fit` — Ajuster automatiquement les ressources

```bash
--fit on    # Ajuste automatiquement les paramètres si la VRAM est insuffisante
```

Désactive les options qui causent des OOM au démarrage. Très pratique pour ne pas avoir à ajuster manuellement.

---

## Paramètres de performance

### `-fa, --flash-attn` — Flash Attention

```bash
-fa on    # Active Flash Attention
```

**Flash Attention** est une optimisation algorithmique de l'attention qui :
- Réduit la complexité mémoire de O(n²) à O(n)
- Accélère le traitement des longs contextes
- Est **indispensable** pour les contextes > 32k tokens
- Compatible uniquement avec les GPU NVIDIA (CUDA)

**Gain typique** : 20-40% plus rapide sur les longs contextes, réduction significative de la VRAM.

### `-t, --threads <n>` — Nombre de threads CPU

```bash
-t 8    # 8 threads CPU pour le traitement hors-GPU
```

Même avec un GPU, les parties non offloadées utilisent le CPU. Le nombre optimal correspond généralement au nombre de cœurs physiques.

### `-b, --batch-size <n>` — Taille de batch pour le préremplissage

```bash
-b 512   # Batch de 512 tokens pour le prompt processing
```

Une taille plus grande accélère le préremplissage mais consomme plus de VRAM temporairement.

---

## Fonctionnement interne

### Cycle de vie d'une requête

```
1. Client envoie POST /v1/chat/completions
2. llama-server tokenize le prompt
3. Prompt processing (préremplissage) :
   - Tous les tokens du prompt sont traités en parallèle
   - Le cache KV est rempli
   - Flash Attention accélère cette étape
4. Génération token par token :
   - Le modèle sélectionne les experts MoE actifs pour ce token
   - Le cache KV est consulté pour le contexte
   - Un token est généré
5. Le token est renvoyé (streaming ou non)
6. Retour à l'étape 4 jusqu'à fin de génération
```

### Mémoire VRAM — Comment c'est réparti

Sur une RTX 4070 12GB avec Qwen3.6-35B-A3B (Q4_K_M) :

```
VRAM totale : 12 GB
├── Poids du modèle : ~6.5 GB
│   ├── Couches principales (GPU) : ~4 GB
│   └── Experts MoE (25 sur GPU) : ~2.5 GB
├── Cache KV (Q8_0, 128k) : ~4-8 GB (croît avec le contexte)
└── Overhead CUDA : ~0.5 GB
```

> Le cache KV croît dynamiquement avec la conversation. C'est pourquoi la VRAM varie entre 6.5 et 10.6 GB.

### Les experts MoE en détail

Un modèle MoE à 35B paramètres avec 64 experts n'active que quelques experts (8 dans le cas de Qwen3.6) par token :

```
Token "Bonjour" → Expert sélectionné : [3, 17, 22, 31, 45, 51, 58, 61]
Token "le"      → Expert sélectionné : [3, 12, 22, 31, 40, 45, 55, 58]
Token "monde"   → Expert sélectionné : [5, 17, 22, 31, 45, 50, 58, 63]
                     ↑                    ↑
                     Experts fréquents    Experts spécifiques au contexte
```

**L'avantage MoE** : Seuls les experts activés sont calculés → 35B params mais seulement ~8B calculés par token (A3B = Active 3B).

---

## Cas pratique : Qwen3.6-35B-A3B

### Commande recommandée (RTX 4070 12GB)

```bash
llama-server \
  -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen35b \
  --host 0.0.0.0 --port 8081 \
  -ngl 999 -ncmoe 25 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 131072 -t 8 \
  --no-context-shift \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.00 --presence-penalty 0.00 \
  --fit on \
  --chat-template-kwargs '{"preserve_thinking": true}'
```

### Explication de chaque flag

| Flag | Pourquoi |
|------|----------|
| `-ngl 999` | Toutes les couches sur GPU pour max perfs |
| `-ncmoe 25` | 25 experts MoE en VRAM — sweet spot pour 12GB |
| `-fa on` | Flash Attention indispensable pour 128k |
| `--cache-type-k/v q8_0` | Cache KV en Q8 — divise par 2 la VRAM, qualité préservée |
| `-c 131072` | Contexte 128k complet |
| `-t 8` | 8 threads CPU pour les parties non-GPU |
| `--no-context-shift` | Pas de glissement silencieux du contexte |
| `--temp 0.6` | Créativité modérée, bon pour le coding |
| `--top-p 0.95 --top-k 20` | Sampling conservateur |
| `--repeat-penalty 1.00` | Pas de pénalité — le MoE gère sa structure |
| `--presence-penalty 0.00` | Pas de biais de diversité |
| `--fit on` | Ajustement auto si VRAM insuffisante |
| `--chat-template-kwargs '{"preserve_thinking": true}'` | Préserve les blocs de réflexion |

### Résultats attendus

| Métrique | Valeur |
|----------|--------|
| Vitesse de génération | 58-62 tok/s |
| VRAM (contexte vide) | ~6.5 GB |
| VRAM (contexte plein 128k) | ~10.6 GB |
| Prompt processing | ~2000 tok/s |

---

## Dépannage

### OOM (Out of Memory) CUDA

```
CUDA error: out of memory
```

Solutions (dans l'ordre) :
1. Réduire `-ncmoe` (ex: 15 au lieu de 25)
2. Réduire `-c` (ex: 65536 au lieu de 131072)
3. Passer le cache KV en `q4_0` (dégradation de qualité)
4. Réduire `-ngl` (ex: 80 au lieu de 999)

### Modèle lent

- Vérifier que `-fa on` est activé
- Vérifier que `-ngl 999` est utilisé
- Vérifier les threads CPU : `-t` doit correspondre aux cœurs physiques
- Vérifier que CUDA est bien utilisé : `nvidia-smi` doit montrer l'utilisation GPU

### Contexte tronqué

- Vérifier `-c 131072` dans la commande
- Vérifier `--no-context-shift` pour éviter les coupures silencieuses
- Augmenter `--cache-type` si la qualité se dégrade en fin de contexte