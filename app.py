"""CTG Signal Analyzer — chef d'orchestre.   Lancer :  streamlit run app.py

Ce fichier charge les données (N'IMPORTE QUEL fichier déposé par l'utilisateur), construit le « contexte »
et le donne à chacune des 4 parties (un onglet = un membre de l'équipe), puis affiche une synthèse.
"""
import pandas as pd
import streamlit as st

import commun as C
import partie1_acquisition as P1
import partie2_frequentiel as P2
import partie3_correlations as P3
import partie4_tempsfreq_ml as P4

st.set_page_config(page_title="CTG Signal Analyzer", page_icon="👶", layout="wide")
C.appliquer_theme()
C.entete("👶 CTG Signal Analyzer",
         "Analyse du rythme cardiaque fœtal et des contractions utérines par Signaux et Systèmes")
st.info("Outil pédagogique d'analyse du signal : il ne constitue pas un dispositif médical et ne remplace pas l'avis d'un professionnel de santé.")

# --------------------------------------------------------------------------- Données
st.sidebar.header("1 · Données")
source = st.sidebar.radio("Source", ["Importer un fichier", "Enregistrement simulé"], label_visibility="collapsed")
AUCUNE, INDEX = "(aucune)", "(index des échantillons)"
if source == "Importer un fichier":
    fichiers = st.sidebar.file_uploader(
        "Déposez n'importe quel enregistrement : CSV, TXT, Excel, ou PhysioNet (.hea + .dat ensemble)",
        type=["csv", "txt", "tsv", "dat", "hea", "xlsx", "xls"], accept_multiple_files=True)
    if not fichiers:
        st.write("👈 Déposez un enregistrement CTG pour lancer l'analyse. Les colonnes (temps, FHR, UC), le séparateur, "
                 "l'unité du temps et la virgule décimale sont reconnus automatiquement.")
        st.stop()
    try:
        df, nom = C.lire_fichiers(fichiers)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
else:
    df, nom = C.generer_signal_demo(), "simulation"
    st.sidebar.caption("Données **simulées** : à indiquer comme telles dans le dossier.")

st.sidebar.header("2 · Colonnes")
g_t, g_f, g_u = C.deviner_colonnes(df)
cols = list(df.columns)
col_t = st.sidebar.selectbox("Temps", [INDEX] + cols, index=cols.index(g_t) + 1 if g_t else 0)
col_f = st.sidebar.selectbox("Rythme cardiaque (FHR)", cols, index=cols.index(g_f) if g_f else 0)
col_u = st.sidebar.selectbox("Contractions (UC)", [AUCUNE] + cols, index=cols.index(g_u) + 1 if g_u else 0)
unite = st.sidebar.selectbox("Unité du temps", ["auto", "s", "min", "ms"])
fs_def = st.sidebar.number_input("Fréquence d'échantillonnage si pas de colonne temps (Hz)", 0.1, 100.0, 4.0, 0.5)

try:
    ctx = C.construire_contexte(df, None if col_t == INDEX else col_t, col_f, None if col_u == AUCUNE else col_u,
                                nom, unite_temps=unite, fs_defaut=fs_def)
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
for a in ctx.avertissements:
    st.warning(a)


def executer(fonction):
    """Une partie en panne n'empêche pas les autres de s'afficher."""
    try:
        fonction(ctx)
    except Exception as e:
        st.error(f"Cette partie a rencontré un problème : {e}")
        with st.expander("Détails techniques"):
            st.exception(e)


# --------------------------------------------------------------------------- Synthèse
def synthese(ctx):
    C.bandeau_partie("🌟", "Synthèse de l'analyse", "Toute l'équipe",
                     "Les résultats clés des 4 parties, et la cohérence entre des méthodes indépendantes.")
    r1, r2, r3 = P1.resultats(ctx), P2.resultats(ctx), P3.resultats(ctx)
    c = st.columns(5)
    c[0].metric("Durée", f"{ctx.duree_s / 60:.1f} min")
    c[1].metric("Signal FHR disponible", f"{r1['disponibilite']:.1f} %")
    c[2].metric("Ligne de base", f"{r1['baseline']:.0f} BPM")
    c[3].metric("Accél. / décél.", f"{r1['n_acc']} / {r1['n_dec']}")
    c[4].metric("Contractions", r1["n_contr"] if ctx.uc is not None else "—")
    st.subheader("Indicateurs usuels d'un CTG (mesurés par traitement du signal)")
    st.dataframe(pd.DataFrame(r1["indicateurs"], columns=["Indicateur", "Mesure", "Repère usuel", "Position"]),
                 width="stretch", hide_index=True)
    st.caption("Mesures comparées à des repères usuels (FIGO 2015) : ce n'est pas un diagnostic et cela ne remplace pas "
               "l'avis d'un professionnel de santé.")
    if ctx.uc is None:
        st.info("Sans signal UC, la synthèse se limite au FHR : pas de rythme des contractions ni de décalage FHR/UC.")
        return
    st.subheader("Période des contractions : 3 méthodes indépendantes")
    periodes = {"FFT de l'UC (partie 2)": 1 / r2["f_dom_uc"] if r2["f_dom_uc"] else None,
                "Autocorrélation de l'UC (partie 3)": r3["periode_auto_uc_s"],
                "Détection de pics (partie 1)": r1["periode_contr_s"]}
    tab = pd.DataFrame({"Méthode": list(periodes),
                        "Période (min)": [None if v is None else round(v / 60, 2) for v in periodes.values()]})
    st.dataframe(tab, width="stretch", hide_index=True)
    vals = [v for v in periodes.values() if v]
    if len(vals) >= 2:
        ecart = (max(vals) - min(vals)) / (sum(vals) / len(vals))
        if ecart <= 0.15:
            st.success(f"Les méthodes s'accordent à {100 * ecart:.0f} % près : la période des contractions est bien établie "
                       f"(≈ {sum(vals) / len(vals) / 60:.1f} min).")
        else:
            st.warning(f"Les méthodes diffèrent de {100 * ecart:.0f} % : contractions irrégulières ou seuils à régler (partie 1).")
    st.subheader("Relation FHR / UC")
    if r3["tau_s"] is not None:
        sens = "baisse" if r3["coef"] < 0 else "monte"
        ordre = "après" if r3["tau_s"] > 0 else "avant"
        st.markdown(f"- Corrélation croisée : coefficient **{r3['coef']:+.2f}**, le FHR {sens} quand l'UC augmente, "
                    f"environ **{abs(r3['tau_s']):.0f} s {ordre}** l'UC.")
    dec = r1["decalages_dec"]
    if dec:
        st.markdown(f"- {len(dec)} décélération(s) associées à une contraction, décalage moyen **{sum(dec) / len(dec):+.0f} s** "
                    "(mesure indépendante, par détection d'événements).")
    st.caption("Une relation temporelle n'est pas une cause, et aucun de ces résultats n'est un diagnostic.")


# --------------------------------------------------------------------------- Onglets
onglets = st.tabs(["🌟 Synthèse", "🍼 ① Acquisition & événements", "🌊 ② Fréquentiel", "🔗 ③ Corrélations & système",
                   "🧸 ④ Temps-fréquence & anomalies", "👥 Équipe"])
with onglets[0]:
    executer(synthese)
for onglet, partie in zip(onglets[1:5], (P1, P2, P3, P4)):
    with onglet:
        executer(partie.afficher)

with onglets[5]:
    C.bandeau_partie("•", "Répartition et fil conducteur", "Équipe", "Chaque membre possède un fichier et une partie de Signaux et Systèmes.")
    st.markdown("""
| Membre | Fichier | Signaux et Systèmes | À expliquer à l'oral |
|---|---|---|---|
| **1** | `partie1_acquisition.py` | Signal discret, échantillonnage et **repliement (Shannon)**, ligne de base, détection de pics | Pourquoi sous-échantillonner sans filtre fait « replier » l'énergie ? Pourquoi médiane et pas filtre ? |
| **2** | `partie2_frequentiel.py` | FFT, fenêtre de Hann, **Parseval**, **DSP (Welch)**, STFT | Que signifie la fréquence dominante de l'UC ? Pourquoi Welch plutôt qu'une FFT ? |
| **3** | `partie3_correlations.py` | Autocorrélation, corrélation croisée, **cohérence et fonction de transfert** | Qu'est-ce que la cohérence ? Pourquoi corrélation ≠ cause ? |
| **4** | `partie4_tempsfreq_ml.py` | Ondelettes (CWT), énergie, caractéristiques | Différence CWT / STFT ? Que fait (et ne fait pas) l'Isolation Forest ? |

**Fil conducteur** : importer → vérifier la qualité → détecter les événements → analyser les fréquences →
modéliser la relation UC → FHR → résumer.
""")
