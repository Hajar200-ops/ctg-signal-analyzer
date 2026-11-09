# 👶 CTG Signal Analyzer

Analyse du rythme cardiaque fœtal (FHR) et des contractions utérines (UC) par Signaux et Systèmes.
Outil pédagogique : il mesure des grandeurs du signal, il ne pose pas de diagnostic.

## Lancer en local
```
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```
pip install pytest
pytest -v
```

## Mettre en ligne (Streamlit Community Cloud)
1. Créer un dépôt GitHub public (ex. `ctg-signal-analyzer`), sans README.
2. Dans le dossier du projet :
```
git init
git add app.py commun.py partie1_acquisition.py partie2_frequentiel.py partie3_correlations.py partie4_tempsfreq_ml.py requirements.txt README.md test_projet.py .gitignore .streamlit
git commit -m "CTG Signal Analyzer"
git branch -M main
git remote add origin https://github.com/VOTRE_NOM/ctg-signal-analyzer.git
git push -u origin main
```
3. Sur https://share.streamlit.io : se connecter avec GitHub, « New app », choisir le dépôt, la branche `main`,
   « Main file path » = `app.py`, « Advanced settings » : Python 3.12, puis « Deploy ».
4. Pour mettre à jour l'application : `git add`, `git commit`, `git push` (le site se met à jour tout seul).

## Fichiers
| Fichier | Rôle |
|---|---|
| `app.py` | dépôt du fichier, choix des colonnes, onglets, synthèse |
| `commun.py` | lecture CSV / TXT / Excel / PhysioNet (.hea + .dat), contexte de données, thème |
| `partie1_acquisition.py` | qualité, interruptions, événements, indicateurs usuels |
| `partie2_frequentiel.py` | FFT, DSP (Welch), STFT |
| `partie3_correlations.py` | autocorrélation, corrélation croisée, cohérence, fonction de transfert, réponse impulsionnelle |
| `partie4_tempsfreq_ml.py` | ondelettes, caractéristiques par fenêtres, Isolation Forest |
| `test_projet.py` | tests par membre |
