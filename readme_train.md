# Document Technique pour le Modèle Prédictif de la Gravité des Accidents

## Introduction
Ce document explique les étapes suivies pour construire un modèle prédictif de la gravité des accidents en utilisant des données provenant de diverses sources. Le but du modèle est de prédire la gravité des accidents en fonction de différents facteurs.

## Sources de Données
quatre dataset ont été utilisé principaux pour entraîner le modèle, chacun contenant différents aspects des accidents de la route en France :
1. **Caractéristiques** : Circonstances générales de l'accident.
2. **Lieux** : Détails de localisation de l'accident.
3. **Véhicules** : Informations sur les véhicules impliqués.
4. **Usagers** : Informations sur les usagers impliqués (conducteurs, passagers, piétons).

## Prétraitement des Données

1. **Chargement des Données** :
   - Les dataset ont été mergés en utilisant un identifiant commun `Num_Acc` pour créer un dataset complet.

2. **Sélection des Variables** :
   -  Des variables pertinentes(feature data) ont été selectionnées telles que les conditions d'éclairage, le type de route, la catégorie de véhicule, et les caractéristiques démographiques des usagers. La variable cible(target data) est la gravité de l'accident (grav), classée en quatre catégories : indemne, tué, hospitalisé, et blessé léger.

3. **Gestion des Valeurs Manquantes** :
   - Les valeurs manquantes dans les variables ont été remplacées par des zéros pour garantir que le modèle puisse traiter les données sans problème.

4. **Encodage des Variables Catégorielles** :
   - Les variables catégorielles (par exemple, type de collision, conditions météorologiques) ont été converties en valeurs numériques en utilisant l'encodage one-hot, un processus qui transforme les catégories en valeurs numériques.

5. **Normalisation des Données** :
   - Les données ont été normalisées pour s'assurer que toutes les variables ont la même échelle, ce qui aide les algorithmes d'apprentissage automatique à les traiter plus efficacement.

6. **Application de la PCA** :
   - L'Analyse en Composantes Principales (PCA) a été appliquée pour réduire la dimensionnalité des données tout en conservant 95% de la variance. Cette étape simplifie le jeu de données en réduisant le nombre de variables.

7. **Gestion du Déséquilibre des Classes** :
   - Le jeu de données a été rééchantillonné en utilisant la technique SMOTE (Synthetic Minority Over-sampling Technique) pour traiter le déséquilibre des classes cibles. Cette technique génère des exemples synthétiques pour les classes minoritaires afin de garantir que le modèle soit entraîné sur un jeu de données équilibré.

## Construction du Modèle

un modèle de réseau de neurones a été construit pour prédire la gravité des accidents. L'architecture du modèle comprend :
1. **Couche d'Entrée** : Accepte les caractéristiques prétraitées.
2. **Couches Cachées** : Trois couches cachées avec des fonctions d'activation ReLU et un dropout pour régulariser et prévenir le surapprentissage.
3. **Couche de Sortie** : Prédit la probabilité de chaque classe de gravité.

Le modèle a été entraîné en utilisant la fonction de perte cross-entropy, une fonction de perte courante pour les tâches de classification, et optimisé à l'aide de l'optimiseur Adam.

## Entraînement et Évaluation

1. **Validation Croisée** :
   - la validation croisée à 5 plis pour entraîner et évaluer le modèle. Cette méthode divise le jeu de données en cinq parties, entraîne le modèle sur quatre parties, et le teste sur la partie restante, en tournant à travers toutes les parties. Cela aide à garantir que le modèle est robuste et se généralise bien aux nouvelles données.
   - L'exactitude de chaque pli a été enregistrée pour mesurer les performances.

2. **Early Stopping** :
   - Pendant l'entraînement, l'entraînement est arreté si ela loss de validation ne s'améliorait pas pendant plusieurs époques. Cette technique, connue sous le nom de early stopping, empêche le surapprentissage et garantit que le modèle n'apprend pas le bruit dans les données d'entraînement.

3. **Entraînement Final du Modèle** :
   - Après la validation croisée, le modèle a été entraîné sur l'ensemble du jeu de données pour maximiser la quantité de données utilisées pour l'apprentissage.

## Résultats

- **Exactitude** :
  - L'exactitude moyenne sur tous les plis pendant la validation croisée a été enregistrée, fournissant une mesure de la performance du modèle sur des données non vues.
- **Rapport de Classification** :
  - Un rapport détaillé montrant la précision, le rappel et le score F1 pour chaque classe.
- **Matrice de Confusion** :
  - Une matrice montrant les classifications vraies vs. prédites, donnant des informations sur les classes qui sont plus susceptibles d'être confondues.

## Prédictions et Données Utilisées

- **Prédictions** :
  - Le modèle prédit la gravité des accidents en quatre catégories : indemne, tué, hospitalisé, et blessé léger. Ces prédictions peuvent aider à comprendre les facteurs menant à des accidents graves et à développer de meilleures mesures de sécurité.
- **Données Utilisées** :
  - Facteurs tels que les conditions d'éclairage, le type de route, la catégorie de véhicule, les caractéristiques démographiques des usagers (par exemple, âge, sexe), les conditions météorologiques, et les conditions de la route.

## Instructions pour Lancer le Code

1. **Installation des Librairies** :
   - Assurez-vous d'avoir Python installé sur votre machine.
   - importez le fichier `requirements.txt` 
   - Installez les librairies en exécutant la commande :
     ```
     pip install -r requirements.txt
     ```

2. **Exécution du Script** :
   - Assurez-vous que le script Python et les fichiers de données sont dans le même répertoire.
   - Exécutez le script principal avec la commande suivante dans le terminal :
     ```
     python main.py
     ```
