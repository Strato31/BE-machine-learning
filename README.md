# BE-ML
BE de machine learning avec Benoît
Les fichiers plus lourds de données ne sont pas sur le git, il faudra les chercher en local.

Le data contient des échantillons de 28 caractéristiques :
- les 21 premières sont des propriétés kinétiques mesurées en accélérateur de particules
- les 7 dernières sont des fonctions haut niveau dérivées par les physiciens pour aider à la classification

Les données contiennent une colonne "label".
Lorsque label=0, les données de l'échantillon décrivent un phénomène de fond, de "bruit"
Lorsque label=1, les données de l'échantillon décrivent un signal produit par un boson de Higgs.

Le problème à résoudre est le suivant : discriminer un échantillon pour savoir s'il est produit par un boson de Higgs ou simplemement un phénomène de fond. C'est un problème de classification.

Mots-clés utilisés : classification, feature/target, feature engineering, corrélation, hyperparamètre, cross-validation, training-testing dataset