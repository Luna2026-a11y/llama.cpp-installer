🇫🇷 **Français** | 🇬🇧 [English](DOC.md)

# Documentation llama.cpp — Paramètres & Internes

> Référence complète pour les options en ligne de commande de `llama-server` et le moteur `llama.cpp`.

---

## Table des matières

1. [Architecture Générale](#architecture-générale)
2. [Paramètres du Modèle](#paramètres-du-modèle)
3. [Paramètres de Contexte](#paramètres-de-contexte)
4. [Paramètres GPU / CUDA](#paramètres-gpu--cuda)
5. [Paramètres MoE (Mixture of Experts)](#paramètres-moe-mixture-of-experts)
6. [Paramètres du KV Cache](#paramètres-du-kv-cache)
7. [Paramètres d'Échantillonnage / Génération](#paramètres-déchantillonnage--génération)
8. [Paramètres du Serveur](#paramètres-du-serveur)
9. [Paramètres de Performance](#paramètres-de-performance)
10. [Quantisation Avancée](#quantisation-avancée)
11. [Types de KV Cache — Approfondissement](#types-de-kv-cache--approfondissement)
12. [Bureau vs Headless](#bureau-vs-headless)
13. [Mécanismes Internes](#mécanismes-internes)
14. [Exemple Pratique : Qwen3.6-35B-A3B](#exemple-pratique-qwen36-35b-a3b)
15. [Dépannage](#dépannage)

---

## Architecture Générale

### Comment fonctionne llama.cpp

`llama.cpp` est un moteur d'inférence LLM en C/C++ optimisé pour le matériel grand public. Il ne fait aucun entraînement — uniquement de l'inférence (génération de texte).

Flux de données :

```
GGUF file → Load into memory → Per-layer dequantization → Inference → Generated tokens
```

### Deux exécutables principaux

|| Binaire | Rôle |
|---------|-------|
| `llama-server` | Serveur HTTP avec API compatible OpenAI (`/v1/chat/completions`) |
| `llama-cli` | CLI interactif pour le chat en terminal |

### Le Format GGUF

GGUF (GPT-Generated Unified Format) est le format de modèle de llama.cpp. Il contient :
- Les poids du modèle (quantisés en Q4, Q5, Q8, etc.)
- Des métadonnées (tokenizer, hyperparamètres, template de chat)
- Des tenseurs organisés par couche

Niveaux de quantisation courants :

| Quant | Taille approx. | Qualité | Usage recommandé |
|-------|---------------|---------|------------------|
| `Q3_K_M` | Le plus petit | Bonne | RAM très limitée |
| `IQ4_XS` | ~35 % du FP16 | Bonne | Économie maximale de taille avec imatrix |
| `Q4_K_M` | ~40 % du FP16 | Très bonne | Point d'équilibre pour GPUs 12 Go |
| `Q4_K_XL` | Légèrement plus grand | Meilleure | GPUs avec plus de VRAM |
| `Q5_K_M` | ~55 % du FP16 | Excellente | Quand la VRAM le permet |
| `Q6_K` | ~65 % du FP16 | Quasi-identique | Quasiment sans perte |
| `Q8_0` | ~75 % du FP16 | Quasi-parfaite | Quantisation virtuellement transparente |
| `UD-Q4_K_XL` | Variable | Unsloth Dynamic | Quantisation adaptative (meilleur rapport taille/qualité) |

> **Note** : Les quantisations `UD-*` (Unsloth Dynamic) ajustent la précision par tenseur, offrant de meilleurs rapports qualité/taille que les quantisations uniformes. Voir [Quantisation Avancée](#quantisation-avancée) pour plus de détails.

---

## Paramètres du Modèle

### `-m, --model <path>` — Chemin vers le fichier GGUF

Chemin vers le fichier modèle quantisé.

```bash
-m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### `--alias <name>` — Alias du modèle

Nom d'affichage utilisé dans l'API et les logs. Pratique quand on exécute plusieurs modèles.

```bash
--alias qwen35b
```

Dans l'API, le modèle sera référencé comme `qwen35b` au lieu du chemin complet.

---

## Paramètres de Contexte

### `-c, --ctx-size <n>` — Taille du contexte (tokens)

Nombre maximum de tokens que le modèle peut « voir » en mémoire. C'est la fenêtre de contexte.

```bash
-c 131072   # 128k tokens
-c 32768    # 32k tokens
-c 8192     # 8k tokens (par défaut)
```

**Impact** :
- Plus de contexte = plus de VRAM/RAM consommée par le KV cache
- 128k contexte = ~8 Go de KV cache en q8_0 pour un MoE 35B
- La vitesse de traitement du prompt diminue à mesure que la taille du contexte augmente

### `-n, --predict <n>` — Maximum de tokens à générer

Limite supérieure pour la génération. Le modèle peut s'arrêter plus tôt s'il rencontre un token de fin.

```bash
-n 32768    # Générer jusqu'à 32k tokens
```

### `--no-context-shift` — Désactiver le décalage de contexte

Par défaut, quand le contexte est plein, llama.cpp peut faire glisser la fenêtre (en supprimant les tokens les plus anciens) pour poursuivre la conversation. Ce flag désactive ce comportement.

**Quand l'utiliser** : Quand vous voulez que le modèle respecte strictement la taille de contexte définie, sans tronquer silencieusement l'historique.

---

## Paramètres GPU / CUDA

### `-ngl, --n-gpu-layers <n>` — Nombre de couches sur le GPU

Définit combien de couches du modèle sont déplacées vers la VRAM du GPU. `999` = toutes les couches.

```bash
-ngl 999    # Tout sur le GPU (recommandé si la VRAM le permet)
-ngl 0      # Tout sur le CPU (très lent)
-ngl 20     # Seulement 20 couches sur le GPU (Offloading partiel)
```

**Impact** :
- Plus de couches sur le GPU = plus rapide, mais plus de VRAM nécessaire
- Si tout tient dans la VRAM, utilisez `-ngl 999`
- Si la VRAM est insuffisante, réduisez progressivement jusqu'à ce que ça tienne

### `-sm, --split-mode <mode>` — Mode de répartition multi-GPU

| Mode | Description |
|------|-------------|
| `row` | Répartition par lignes (par défaut) — équilibre entre les GPUs |
| `layer` | Répartition par couches — chaque GPU gère des couches entières |

Utile uniquement avec plusieurs GPUs NVIDIA.

---

## Paramètres MoE (Mixture of Experts)

### `-ncmoe, --n-cpu-moe <n>` — Nombre de couches MoE à garder sur le CPU

**Paramètre clé pour les modèles MoE avec une VRAM limitée.**

Les modèles MoE (comme Qwen3.6-35B-A3B, Mixtral, DeepSeek) ont plusieurs « experts » par couche, mais n'en activent que quelques-uns par token. Le défi : tous les experts doivent être en mémoire pour sélectionner lesquels activer.

Le flag `-ncmoe` contrôle combien de **couches MoE** restent sur le **CPU** (déchargées du GPU). Plus la valeur est élevée = plus de couches MoE sur le CPU = moins de VRAM utilisée, mais plus lent :

```bash
-ncmoe 0     # Toutes les couches MoE sur le GPU — le plus rapide, mais nécessite plus de VRAM
-ncmoe 25    # 25 couches MoE sur le CPU — point d'équilibre pour 12 Go de VRAM
-ncmoe 999   # Toutes les couches MoE sur le CPU (équivalent à --cpu-moe)
```

> **Note** : La forme longue est `--n-cpu-moe`, ce qui rend le sens clair : N couches de poids MoE gardées sur le **CPU**.

**Comment ça fonctionne** :

1. Le modèle MoE a N experts par couche (ex. 64 pour Qwen3.6)
2. À chaque token, seuls quelques experts sont activés (ex. 8 sur 64)
3. Mais TOUS les experts doivent être accessibles pour la sélection
4. `-ncmoe 25` garde les 25 premières couches MoE sur le CPU, le reste sur le GPU
5. Cela réduit l'utilisation de VRAM au prix d'un accès plus lent aux experts déchargés

**Impact sur les performances** :

| Valeur | VRAM | Vitesse | Recommandation |
|--------|------|---------|----------------|
| `-ncmoe 0` | ~10,5 Go | 60-65 tok/s | Tient dans 12 Go de VRAM confortablement ★ |
| `-ncmoe 25` | ~6,5-10 Go | 50-58 tok/s | 12 Go de VRAM serrés / marge de sécurité |
| `-ncmoe 999` / `--cpu-moe` | ~4 Go | 25-30 tok/s | GPU très limité |

> **Astuce** : Avec `--fit on`, llama-server ajustera automatiquement le déchargement MoE si la VRAM est insuffisante. Commencez avec `-ncmoe 0` et laissez `--fit on` gérer le reste.

---

## Paramètres du KV Cache

### `-ctk, --cache-type-k <type>` — Type de quantisation du KV cache (clés)

### `-ctv, --cache-type-v <type>` — Type de quantisation du KV cache (valeurs)

Le KV cache (Key-Value) stocke les représentations intermédiaires des tokens déjà traités, pour éviter de les recalculer à chaque nouveau token.

```bash
-ctk q8_0   # Clés quantisées en Q8 (8 bits)
-ctv q8_0   # Valeurs quantisées en Q8 (8 bits)
```

**Types disponibles** :

| Type | Précision | VRAM | Cohérence en contexte long |
|------|-----------|------|---------------------------|
| `f16` | Complète | 2x | Référence |
| `q8_0` | 8 bits | 1x | Très bonne ★ |
| `q4_0` | 4 bits | 0,5x | Dégradation notable |
| `q4_1` | 4 bits améliorée | 0,5x | Acceptable |

**Pourquoi q8_0 est le point d'équilibre par défaut** :
- Divise par deux la VRAM du KV cache par rapport au FP16
- Préserve suffisamment de précision pour les contextes longs (128k)
- La dégradation est imperceptible en pratique
- q4_0 est trop agressif — le modèle perd en cohéquence au-delà de 32k tokens

**Impact sur un contexte de 128k** :
- f16 : ~16 Go de KV cache → impossible avec 12 Go de VRAM
- q8_0 : ~8 Go de KV cache → tient dans 12 Go de VRAM avec une marge
- q4_0 : ~4 Go mais qualité dégradée

> **Voir aussi** : La section [Types de KV Cache — Approfondissement](#types-de-kv-cache--approfondissement) pour une analyse détaillée de chaque type de cache, y compris `turbo3` et les considérations spécifiques au bureau.

---

## Paramètres d'Échantillonnage / Génération

### `--temp <value>` — Température

Contrôle la « créativité » du modèle. Plus bas = plus déterministe.

```bash
--temp 0.6   # Bon pour le code — créatif mais cohérent
--temp 0.0   # Décodage glouton — toujours le même résultat
--temp 1.0   # Très aléatoire
```

**Valeurs recommandées** :
- Code/analyse : 0,3 - 0,6
- Chat créatif : 0,7 - 0,9
- Tâches déterministes : 0,0

### `--top-p <value>` — Échantillonnage par noyau (nucleus sampling)

Ne considère que les tokens dont la probabilité cumulée atteint ce seuil.

```bash
--top-p 0.95   # Tokens couvrant 95 % de la probabilité
```

### `--top-k <value>` — Échantillonnage Top-K

Ne considère que les K tokens les plus probables.

```bash
--top-k 20   # Seulement les 20 tokens les plus probables
```

### `--repeat-penalty <value>` — Pénalité de répétition

Augmente la pénalité pour les tokens déjà générés, empêchant les boucles.

```bash
--repeat-penalty 1.00   # Pas de pénalité (recommandé pour MoE avec thinking)
```

> **Note** : Pour les modèles avec un mode « thinking » (comme Qwen3), une pénalité de répétition de 1,0 est recommandée car le modèle gère naturellement sa structure de pensée.

### `--presence-penalty <value>` — Pénalité de présence

Pénalise les tokens déjà apparus, encourageant la diversité.

```bash
--presence-penalty 0.00   # Pas de pénalité supplémentaire
```

### `--chat-template-kwargs` — Arguments du template de chat

Paramètres supplémentaires passés au template de chat du modèle.

```bash
--chat-template-kwargs '{"preserve_thinking": true}'
```

`preserve_thinking: true` est **essentiel** pour les modèles avec un mode de pensée/raisonnement (comme Qwen3). Sans cela, le contenu de pensée (`<think>...</think>`) est silencieusement retiré des réponses de l'API.

---

## Paramètres du Serveur

### `--host <address>` — Adresse d'écoute

```bash
--host 0.0.0.0   # Écouter sur toutes les interfaces (accès réseau)
--host 127.0.0.1  # Local uniquement
```

### `--port <port>` — Port d'écoute

```bash
--port 8081    # Port personnalisé
--port 8001    # Autre port
```

### `--fit` — Ajustement automatique des ressources

```bash
--fit on    # Ajuste automatiquement les paramètres si la VRAM est insuffisante
```

Désactive les options qui causent une OOM (Out of Memory) au démarrage. Très pratique pour éviter un réglage manuel.

---

## Paramètres de Performance

### `-fa, --flash-attn` — Flash Attention

```bash
-fa on    # Activer Flash Attention
```

**Flash Attention** est une optimisation algorithmique du mécanisme d'attention qui :
- Réduit la complexité mémoire de O(n²) à O(n)
- Accélère le traitement des contextes longs
- Est **essentielle** pour les contextes > 32k tokens
- Uniquement compatible avec les GPUs NVIDIA (CUDA)

**Gain typique** : 20-40 % plus rapide sur les contextes longs, réduction significative de la VRAM.

### `-t, --threads <n>` — Nombre de threads CPU

```bash
-t 8    # 8 threads CPU pour le traitement hors GPU
```

Même avec un GPU, les parties non déchargées utilisent le CPU. Le nombre optimal correspond généralement au nombre de cœurs physiques.

### `-b, --batch-size <n>` — Taille de batch pour le préremplissage

```bash
-b 512   # Batch de 512 tokens pour le traitement du prompt
```

Des tailles plus grandes accélèrent le préremplissage (prefill) mais consomment temporairement plus de VRAM.

---

## Quantisation Avancée

Au-delà des quantisations standard `Q4_K_M` et `Q5_K_M`, llama.cpp supporte des méthodes de quantisation avancées qui offrent de meilleurs compromis taille/qualité — mais elles comportent des nuances qu'il vaut la peine de comprendre.

### IQ4_XS — Quantisation basée sur l'imatrix

IQ4_XS est une quantisation **basée sur une matrice d'importance (imatrix)** qui répartit les bits par tenseur plutôt que uniformément. Au lieu que chaque tenseur reçoive le même traitement 4 bits, les tenseurs qui importent le plus pour la qualité de sortie reçoivent une précision supérieure, tandis que les tenseurs moins critiques sont quantisés plus agressivement.

**Caractéristiques clés** :
- Plus petit que `Q4_K_M` (~35 % du FP16 contre ~40 %) tout en maintenant une qualité compétitive
- Nécessite une matrice d'importance (imatrix) calculée lors de la quantisation — l'imatrix capture quels tenseurs sont les plus importants pour la qualité d'inférence
- La répartition des bits par tenseur signifie que les couches d'attention et de sortie critiques conservent plus d'information qu'un Q4 uniforme

**L'astuce QKV de cHunter789** : Par défaut, les quantisations IQ4_XS (telles que produites par les scripts de quantisation courants) forcent les couches d'attention QKV (query/key/value) à un minimum de précision `Q5_K`, quel que soit le niveau de quantisation cible. Cela préserve la qualité d'attention mais ajoute ~0,4 Go à la taille du modèle par rapport à ce qu'une distribution stricte par imatrix produirait.

Si vous souhaitez récupérer ces ~0,4 Go (au prix d'une légère perte de qualité dans la précision d'attention), vous pouvez annuler le commit qui a introduit ce seuil QKV. Cette astuce est communément appelée le **cHunter789 QKV override** :

```bash
# When building llama.cpp from source, revert the QKV floor commit
# before compiling, to allow IQ4_XS to quantize QKV layers normally
# rather than forcing Q5_K minimum. Saves ~0.4 GB on 27B-class models.
git revert <QKV-floor-commit-hash>
```

C'est une optimisation avancée — à considérer uniquement si chaque mégaoctet de VRAM compte et que vous pouvez tolérer une attention légèrement plus bruitée.

### Quantisations UD (Unsloth Dynamic)

Les quantisations UD (`UD-Q4_K_XL`, `UD-Q4_K_M`, etc.) utilisent une stratégie de **précision dynamique par tenseur** développée par Unsloth. Plutôt que de choisir un seul niveau de quantisation pour tout le modèle, les quantisations UD :

1. **Profilent l'importance de chaque tenseur** en utilisant des données d'étalonnage
2. **Attribuent une précision supérieure** (ex. Q6_K, Q5_K) aux tenseurs critiques pour l'attention comme QKV et les projections de sortie
3. **Attribuent une précision inférieure** (ex. Q4_K, Q3_K) aux tenseurs moins impactants comme les couches intermédiaires FFN
4. **Optimisent le compromis global taille/qualité** — le résultat surpasse souvent un `Q4_K_M` uniforme à des tailles de fichier similaires ou inférieures

L'effet net est qu'un modèle `UD-Q4_K_XL` a approximativement la même taille qu'un `Q4_K_M` mais avec une perplexité mesurablement meilleure, car les bits sont alloués là où ils comptent le plus.

**Quand préférer UD** :
- Si votre GPU peut accueillir `Q5_K_M` mais pas `Q6_K`, essayez `UD-Q4_K_XL` — qualité similaire dans moins d'espace
- Si vous voulez exploiter au maximum chaque token de contexte, la taille de fichier réduite libère directement de la VRAM pour le KV cache

### Tableau de Comparaison des Quantisations

| Quant | Taille approx. | Qualité | Notes |
|-------|---------------|---------|-------|
| `Q3_K_M` | Le plus petit (~25 % FP16) | Bonne | Dernier recours pour RAM très limitée |
| `IQ4_XS` | ~35 % du FP16 | Bonne | Plus petite basée sur imatrix ; seuil QKV ajoute ~0,4 Go |
| `Q4_K_M` | ~40 % du FP16 | Très bonne | Point d'équilibre standard |
| `Q4_K_XL` | Légèrement plus grand | Meilleure | Plus de marge |
| `Q5_K_M` | ~55 % du FP16 | Excellente | Haute qualité, nécessite plus de VRAM |
| `Q6_K` | ~65 % du FP16 | Quasi-identique | Quasiment sans perte |
| `Q8_0` | ~75 % du FP16 | Quasi-parfaite | Virtuellement transparente |
| `UD-Q4_K_XL` | ~40 % du FP16 | Meilleure que Q4_K_M | Meilleur rapport taille/qualité à ce niveau |
| `UD-Q4_K_M` | ~38 % du FP16 | Similaire à Q4_K_M, plus petit | Précision adaptative par tenseur |
| `UD-Q5_K_M` | ~55 % du FP16 | Meilleure que Q5_K_M | Précision adaptative par tenseur |

---

## Types de KV Cache — Approfondissement

Le KV cache est la plus grande variable dans votre budget VRAM après les poids du modèle. Choisir le bon type de cache détermine la quantité de contexte que vous pouvez accommoder et la vitesse d'inférence en profondeur. Cette section va au-delà du tableau de référence rapide dans [Paramètres du KV Cache](#paramètres-du-kv-cache) et couvre les compromis pratiques.

### turbo3 KV — Contexte Maximum

**turbo3** est un type de KV cache hybride qui divise la précision par head dans chaque couche : les heads Q/K utilisent `q8_0` et les heads V utilisent `q4_0`. Cela vous donne nettement plus de marge de contexte que le `q8_0` pur tout en gardant une précision des clés supérieure au `q4_0` pur.

```bash
-ctk q8_0 -ctv turbo3   # Clés en q8_0, valeurs avec répartition hybride turbo3
```

Ou plus couramment les deux en turbo3 :

```bash
-ctk turbo3 -ctv turbo3
```

**Comment ça fonctionne** :
- La clé de chaque head d'attention est stockée en `q8_0` (8 bits)
- La valeur de chaque head d'attention est stockée en `q4_0` (4 bits)
- L'approche hybride par head signifie que les clés — qui déterminent le routage d'attention — restent précises
- Les valeurs — qui sont mélangées lors de l'attention — peuvent tolérer une précision inférieure

**Compromis** : turbo3 maximise la longueur de contexte mais **la vitesse se dégrade en contexte profond**. Les benchmarks sur GPUs NVIDIA Blackwell montrent :

| Profondeur de contexte | Vitesse (KV turbo3) |
|------------------------|---------------------|
| Peu profond (≤8k) | ~46 tok/s |
| Moyen (≈32k) | ~30 tok/s |
| Profond (≈65k) | ~19 tok/s |

La baisse de vitesse est inhérente au cache de valeurs de plus faible précision — sur des séquences plus longues, le bruit de quantisation s'accumule et le mécanisme d'attention fait plus de travail pour compenser.

### q4_0 KV — Point d'équilibre pour Bureau

Si vous exécutez sur un GPU de bureau (pas headless), le KV `q4_0` est souvent le choix pragmatique :

```bash
-ctk q4_0 -ctv q4_0
```

- **Plus de contexte** que `q8_0` — environ le double de la fenêtre de contexte pour la même VRAM
- **Meilleure vitesse en profondeur** que `turbo3` — le 4 bits uniforme est plus simple à traiter par le kernel
- **Qualité acceptable** jusqu'à ~32k tokens ; dégradation notable au-delà

Pour les utilisateurs de bureau avec un GPU de 16 Go, le KV `q4_0` offre souvent le meilleur équilibre entre longueur de contexte et vitesse soutenue dans les longues conversations.

### q8_0 KV — Qualité de Référence

```bash
-ctk q8_0 -ctv q8_0
```

- **La référence de qualité** : différence imperceptible par rapport au f16 pour la plupart des tâches
- **VRAM divisée par deux** par rapport au f16 — c'est la baseline standard
- **Vitesse constante** à toutes les profondeurs de contexte — pas de courbe de dégradation
- **Contexte limité** sur les petits GPUs — sur une carte de 16 Go avec un modèle 27B, vous plafonnerez autour de 55k tokens en mode headless

### f16 KV — Précision Complète

```bash
-ctk f16 -ctv f16
```

Cache fp16 complet. Qualité la plus élevée, coût en VRAM le plus élevé (~2× q8_0). Rarement utilisé en dehors du benchmarking ou de la validation de qualité — sur la plupart des GPUs grand public, la fenêtre de contexte est trop petite pour être pratique.

### Tableau de Comparaison du KV Cache (Modèle Dense 27B)

| Type KV | Coût par token | Limite de contexte (16 Go headless) | Qualité | Vitesse en profondeur |
|---------|---------------|--------------------------------------|---------|----------------------|
| `f16` | ~8 Ko/token | ~27k tokens | Référence | Constante, rapide |
| `q8_0` | ~4 Ko/token | ~55k tokens | Quasi-parfaite | Constante |
| `q4_0` | ~2 Ko/token | ~110k tokens | Bonne jusqu'à ~32k | Bonne |
| `turbo3` | ~4 Ko/token (K) + ~2 Ko/token (V) | ~110k tokens (headless) | Bonne jusqu'à ~65k | Se dégrade en profondeur |

> **Note** : Le coût par token varie selon l'architecture du modèle (nombre de heads, dimension de head). Pour un modèle dense 27B, le KV `q8_0` est d'environ 4 Ko/token. Les modèles MoE avec des couches d'attention partagées peuvent être inférieurs. Vérifiez toujours les logs de démarrage de `llama-server` pour les budgets mémoire exacts.

### Comparaison de Contexte Bureau vs Headless

Le tableau suivant montre les limites de contexte réalistes pour un modèle dense 27B sur un **GPU de 16 Go** (ex. RTX 4080, RTX 5070 Ti). Les chiffres pour le bureau supposent ~500 Mio de surcharge du serveur X et du compositeur.

| Type KV | Contexte Headless | Contexte Bureau |
|---------|-------------------|-----------------|
| `turbo3` / `q4_0` | ~110k tokens | ~75k tokens |
| `q8_0` | ~55k tokens | ~40k tokens |
| `f16` | ~27k tokens | ~20k tokens |

L'écart de ~35k tokens entre headless et bureau pour `turbo3`/`q4_0` provient directement des ~487 Mio perdus pour le serveur d'affichage (voir [Bureau vs Headless](#bureau-vs-headless) ci-dessous).

---

## Bureau vs Headless

### Le Problème de VRAM sur Bureau

**Les chiffres de benchmark headless ne se traduisent pas directement en utilisation bureau.** C'est l'une des sources de confusion les plus courantes lors de la configuration de llama.cpp.

Quand vous exécutez `nvidia-smi` sur un GPU avec 16 Go (16304 Mio) de VRAM sur un bureau Ubuntu, la VRAM réellement disponible pour llama.cpp est significativement inférieure :

```
$ nvidia-smi
# Reports: 16304 MiB total

# But after X server + compositor:
# Actual available: ~15817 MiB
# Overhead:        ~487 MiB
```

Ces ~500 Mio sont consommés par :
- Le serveur d'affichage X.org ou Wayland
- Le compositeur GPU (Mutter, KWin, etc.)
- Tout bureau rendu sur le GPU (même les bureaux inactifs consomment de la VRAM pour les framebuffers)
- Les applications accélérées par GPU (navigateur, terminal, etc.)

**Le calcul** : Sur un modèle dense 27B avec KV `turbo3`, chaque token de contexte coûte environ **4 Ko**. Perdre 487 Mio signifie :

```
487 MiB × 1024 KiB/MiB ÷ 4 KiB/token ≈ 124 672 tokens
```

En pratique, l'impact est d'environ ~35k tokens car la VRAM est répartie entre le KV cache et d'autres allocations, et il y a de la fragmentation. Mais le principe est clair : **sur un bureau, vous perdez environ 3 % de votre capacité de contexte uniquement pour le serveur d'affichage**.

### Comment Vérifier Votre VRAM Disponible

**Méthode 1** — `nvidia-smi` (avant de démarrer llama-server) :

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

Cela montre ce que `nvidia-smi` pense être libre — mais inclut l'allocation du serveur d'affichage.

**Méthode 2** — Vérifier les logs de démarrage de `llama-server` :

Quand `llama-server` démarre, il affiche le budget VRAM que ggml voit :

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080, compute capability 8.9, VMM: yes
  ...
  VRAM budget: 15817 MiB
```

La ligne `VRAM budget` vous indique exactement ce que ggml peut allouer — **c'est le chiffre qui compte**, pas le total de `nvidia-smi`.

Si le budget VRAM de ggml est significativement inférieur à ce que `nvidia-smi` rapporte comme total, la différence est votre surcharge d'affichage.

**Méthode 3** — Exécuter en headless pour confirmer :

```bash
# Stop your display manager temporarily
sudo systemctl stop gdm   # or lightdm, sddm, etc.
# Run llama-server
./llama-server ...
# Check VRAM budget in logs — it should now be close to 16304 MiB
```

### Conseils pour les Utilisateurs de Bureau

- **Utilisez `-ctk q4_0 -ctv q4_0`** ou **`-ctk turbo3 -ctv turbo3`** si vous avez besoin d'un contexte maximal sur un bureau — le coût réduit par token compense la VRAM perdue pour le serveur d'affichage.
- **Utilisez `--fit on`** pour laisser llama-server réduire automatiquement le contexte ou décharger des couches si la VRAM est insuffisante.
- **Fermez les applications accélérées par GPU** (navigateurs avec accélération matérielle, lecteurs vidéo, etc.) avant de démarrer des sessions à contexte long.
- **Envisagez d'exécuter en headless** (sans serveur d'affichage) pour une capacité de contexte maximale, accessible via SSH ou un GPU intégré séparé pour l'affichage.

---

## Mécanismes Internes

### Cycle de Vie d'une Requête

```
1. Client sends POST /v1/chat/completions
2. llama-server tokenizes the prompt
3. Prompt processing (prefill):
   - All prompt tokens are processed in parallel
   - KV cache is populated
   - Flash Attention accelerates this step
4. Token-by-token generation:
   - Model selects active MoE experts for this token
   - KV cache is consulted for context
   - One token is generated
5. Token is returned (streaming or not)
6. Return to step 4 until generation ends
```

### Disposition de la VRAM

Sur une RTX 4070 12 Go avec Qwen3.6-35B-A3B (Q4_K_M) :

```
Total VRAM : 12 GB
├── Model weights : ~6.5 GB
│   ├── Main layers (GPU) : ~4 GB
│   └── MoE experts (all on GPU with -ncmoe 0) : ~2.5 GB
├── KV cache (q8_0, 128k) : ~4-8 GB (grows with context)
└── CUDA overhead : ~0.5 GB
```

> Avec `-ncmoe 0`, toutes les couches MoE sont sur le GPU pour une vitesse maximale. Si la VRAM est limitée, utilisez `-ncmoe 25` pour décharger 25 couches MoE sur le CPU.

### Experts MoE en Détail

Un modèle MoE de 35B paramètres avec 64 experts n'en active que quelques-uns (8 pour Qwen3.6) par token :

```
Token "Hello"  → Selected experts : [3, 17, 22, 31, 45, 51, 58, 61]
Token "world"  → Selected experts : [3, 12, 22, 31, 40, 45, 55, 58]
Token "today"  → Selected experts : [5, 17, 22, 31, 45, 50, 58, 63]
                    ↑                    ↑
                    Frequent experts     Context-specific experts
```

**Avantage MoE** : Seuls les experts activés sont calculés → 35B paramètres mais seulement ~8B calculés par token (A3B = Active 3B).

---

## Exemple Pratique : Qwen3.6-35B-A3B

### Commande Recommandée (RTX 4070 12 Go)

```bash
llama-server \
  -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen35b \
  --host 0.0.0.0 --port 8081 \
  -ngl 999 -ncmoe 0 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 131072 -t 8 \
  --no-context-shift \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.00 --presence-penalty 0.00 \
  --fit on \
  --chat-template-kwargs '{"preserve_thinking": true}'
```

### Explication Flag par Flag

| Flag | Pourquoi |
|------|----------|
| `-ngl 999` | Toutes les couches sur le GPU pour une performance maximale |
| `-ncmoe 0` | Toutes les couches MoE sur le GPU — le plus rapide (utiliser avec `--fit on` pour la sécurité) |
| `-fa on` | Flash Attention essentielle pour 128k |
| `-ctk q8_0 -ctv q8_0` | KV cache Q8 — divise la VRAM par deux, préserve la qualité |
| `-c 131072` | Contexte 128k complet |
| `-t 8` | 8 threads CPU pour les parties hors GPU |
| `--no-context-shift` | Pas de glissement silencieux du contexte |
| `--temp 0.6` | Créativité modérée, bon pour le code |
| `--top-p 0.95 --top-k 20` | Échantillonnage conservateur |
| `--repeat-penalty 1.00` | Pas de pénalité — le MoE gère sa structure |
| `--presence-penalty 0.00` | Pas de biais de diversité |
| `--fit on` | Ajustement automatique si VRAM insuffisante |
| `--chat-template-kwargs '{"preserve_thinking": true}'` | Préserver les blocs de pensée |

### Résultats Attendus

| Métrique | Valeur |
|----------|--------|
| Vitesse de génération | 58-62 tok/s |
| VRAM (contexte vide) | ~6,5 Go |
| VRAM (contexte 128k complet) | ~10,6 Go |
| Traitement du prompt | ~2000 tok/s |

---

## Dépannage

### CUDA OOM (Out of Memory)

```
CUDA error: out of memory
```

Solutions (dans l'ordre) :
1. Augmenter `-ncmoe` (ex. `-ncmoe 25` pour décharger les couches MoE sur le CPU, réduisant la VRAM)
2. Réduire `-c` (ex. `-c 65536` au lieu de `-c 131072`)
3. Passer le KV cache en `q4_0` (dégradation de qualité) : `-ctk q4_0 -ctv q4_0`
4. Réduire `-ngl` (ex. `-ngl 80` au lieu de `-ngl 999`)

### Surcharge VRAM sur Bureau — Le Serveur X Consomme ~500 Mio

Si votre modèle se charge en headless mais fait un OOM sur le bureau, le serveur X et le compositeur consomment de la VRAM que `nvidia-smi` n'indique pas toujours clairement.

**Symptômes** :
- `nvidia-smi` rapporte 16304 Mio au total, mais le budget VRAM de ggml n'est que ~15817 Mio
- Le même réglage `-c` qui fonctionne en headless provoque un OOM sur le bureau
- Vous perdez ~35k tokens de contexte par rapport aux benchmarks headless

**Solutions** :
- Utilisez `-ctk q4_0 -ctv q4_0` ou `-ctk turbo3 -ctv turbo3` pour caser plus de contexte dans moins de VRAM
- Ajoutez `--fit on` pour laisser le serveur s'ajuster automatiquement
- Fermez les applications accélérées par GPU avant de démarrer llama-server
- Exécutez en headless (arrêtez le gestionnaire d'affichage) pour une capacité maximale

### Inférence Lente en Contexte Profond (Dégradation KV turbo3)

Si vous utilisez le KV `turbo3` et remarquez que la vitesse de génération chute significativement à mesure que le contexte se remplit, c'est un comportement attendu.

**Symptômes** :
- Contexte peu profond (≤8k) : ~46 tok/s
- Contexte profond (≈65k) : ~19 tok/s

**Solutions** :
- Passez à `-ctk q4_0 -ctv q4_0` pour une meilleure vitesse en profondeur (moins de capacité de contexte mais vitesse plus constante)
- Passez à `-ctk q8_0 -ctv q8_0` si vous pouvez accepter moins de tokens — la vitesse reste constante
- Réduisez `-c` pour limiter la fenêtre de contexte et garder une vitesse plus élevée

### Modèle Lent (Général)

- Vérifiez que `-fa on` est activé (Flash Attention est critique pour la vitesse)
- Vérifiez que `-ngl 999` est utilisé (toutes les couches sur le GPU)
- Vérifiez les threads CPU : `-t` doit correspondre aux cœurs physiques
- Vérifiez que CUDA est réellement utilisé : `nvidia-smi` doit montrer une utilisation GPU pendant l'inférence
- Vérifiez que votre modèle n'est pas partiellement sur le CPU — cherchez les messages « offloaded » dans les logs de démarrage

### Modèle Introuvable / Mauvais Chemin

```
error: failed to open model file: models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

**Solutions** :
- Utilisez un chemin absolu : `-m /home/user/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Vérifiez que le fichier existe : `ls -la /path/to/model.gguf`
- Cherchez les erreurs de frappe dans le nom du fichier (sensible à la casse sur Linux)
- Assurez-vous que le fichier `.gguf` n'est pas corrompu : vérifiez que la taille du fichier correspond à ce que vous avez téléchargé

### Port Déjà Utilisé

```
error: failed to bind port 8081: Address already in use
```

**Solutions** :
- Vérifiez ce qui utilise le port : `ss -tlnp | grep 8081` ou `lsof -i :8081`
- Utilisez un port différent : `--port 8082`
- Tuez le processus existant si obsolète : `kill <PID>`

### nvidia-smi Affiche le GPU mais la Compilation CUDA Échoue

```
ggml_cuda_init: CUDA not found
```

Ou le binaire `llama-server` s'exécute sans accélération GPU malgré `nvidia-smi` fonctionnel.

**Cause** : `nvidia-smi` provient du pilote NVIDIA, mais compiler llama.cpp avec le support CUDA nécessite le **NVIDIA CUDA Toolkit** (en-têtes de développement + compilateur).

**Solutions** :
- Installez le CUDA toolkit : `sudo apt install nvidia-cuda-toolkit` (Ubuntu) ou téléchargez depuis [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
- Vérifiez que `nvcc` est disponible : `nvcc --version`
- Reconstruisez llama.cpp : le script de compilation devrait détecter CUDA via `nvcc`
- Si vous compilez manuellement, assurez-vous que `-DGGML_CUDA=ON` est défini dans votre configuration CMake

### GPU Blackwell — Note sur les Kernels Flash Attention

Sur les GPUs NVIDIA Blackwell (RTX série 50, B100, B200), le kernel Flash Attention peut nécessiter une version récente de llama.cpp. Les versions plus anciennes peuvent utiliser des chemins d'attention lents ou échouer silencieusement.

**Solutions** :
- Utilisez la dernière version de llama.cpp — le support de Flash Attention pour Blackwell a été ajouté récemment
- Si vous voyez des erreurs référençant `flash_attn` ou des échecs de compilation de kernels, mettez à jour llama.cpp
- Vérifiez avec `-fa on` et cherchez la confirmation `"flash attention"` dans le log de démarrage

### Contexte Tronqué

- Vérifiez `-c 131072` dans la commande
- Vérifiez `--no-context-shift` pour éviter la troncature silencieuse
- Passez à un type de KV cache inférieur (`q4_0` ou `turbo3`) si la qualité se dégrade près des limites de contexte
- Vérifiez votre budget VRAM réel dans les logs de démarrage — vous manquez peut-être de VRAM et ggml réduit silencieusement le contexte