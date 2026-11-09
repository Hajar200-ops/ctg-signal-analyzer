"""CTG Signal Analyzer.   Lancer :  streamlit run app.py"""
import pandas as pd
import streamlit as st

import commun as C
import partie1_acquisition as P1
import partie2_frequentiel as P2
import partie3_correlations as P3
import partie4_tempsfreq_ml as P4

st.set_page_config(page_title="CTG Signal Analyzer", page_icon="👶", layout="wide")
C.appliquer_theme()

st.markdown("""
<div style="
    background: linear-gradient(120deg, #E5F0F3, #FCECEF);
    padding: 28px 32px;
    border-radius: 20px;
    border: 1px solid #D8E7EC;
    margin-bottom: 18px;
">
    <div style="font-size: 14px; color: #66899A; font-weight: 600;">
        MATERNAL & FETAL SIGNAL TECHNOLOGY
    </div>
    <h1 style="color: #263D52; margin: 8px 0;">
        👶 CTG Signal Analyzer
    </h1>
    <p style="font-size: 17px; color: #526D7A; margin-bottom: 0;">
        Plateforme intelligente de visualisation et d'analyse
        des signaux cardiaques fœtaux et des contractions utérines.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Projet pédagogique de traitement du signal. "
)

# --------------------------------------------------------------------------- Dépôt du fichier
fichiers = st.file_uploader("Déposez un enregistrement CTG : CSV, TXT, Excel, ou PhysioNet (.hea + .dat ensemble)",
                            type=["csv", "txt", "tsv", "dat", "hea", "xlsx", "xls"], accept_multiple_files=True)
if not fichiers:
    st.stop()
try:
    df, nom = C.lire_fichiers(fichiers)
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()

AUCUNE, INDEX = "(aucune)", "(index des échantillons)"
st.sidebar.header("Colonnes")
g_t, g_f, g_u = C.deviner_colonnes(df)
cols = list(df.columns)
col_t = st.sidebar.selectbox("Temps", [INDEX] + cols, index=cols.index(g_t) + 1 if g_t else 0)
col_f = st.sidebar.selectbox("Rythme cardiaque (FHR)", cols, index=cols.index(g_f) if g_f else 0)
col_u = st.sidebar.selectbox("Contractions (UC)", [AUCUNE] + cols, index=cols.index(g_u) + 1 if g_u else 0)
unite = st.sidebar.selectbox("Unité du temps", ["auto", "s", "min", "ms"])
fs_def = st.sidebar.number_input("Fréquence d'échantillonnage sans colonne temps (Hz)", 0.1, 100.0, 4.0, 0.5)

try:
    ctx = C.construire_contexte(df, None if col_t == INDEX else col_t, col_f, None if col_u == AUCUNE else col_u,
                                nom, unite_temps=unite, fs_defaut=fs_def)
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
for a in ctx.avertissements:
    st.warning(a)


def executer(fonction):
    """Une section en erreur n'empêche pas les autres de s'afficher."""
    try:
        fonction(ctx)
    except Exception as e:
        st.error(f"Cette section a rencontré un problème : {e}")
        with st.expander("Détails techniques"):
            st.exception(e)


# --------------------------------------------------------------------------- Synthèse
def synthese(ctx):
    r1, r2, r3 = P1.resultats(ctx), P2.resultats(ctx), P3.resultats(ctx)
    st.caption(f"Fichier : {ctx.nom}")
    c = st.columns(4)
    c[0].metric("Durée", f"{ctx.duree_s / 60:.1f} min")
    c[1].metric("Signal FHR disponible", f"{r1['disponibilite']:.1f} %")
    c[2].metric("Ligne de base", f"{r1['baseline']:.0f} BPM")
    c[3].metric("Contractions", r1["n_contr"] if ctx.uc is not None else "—")
    st.subheader("Indicateurs")
    st.dataframe(pd.DataFrame(r1["indicateurs"], columns=["Indicateur", "Mesure", "Repère usuel", "Position"]),
                 width="stretch", hide_index=True)
    if ctx.uc is None:
        return
    st.subheader("Période des contractions")
    periodes = {"FFT de l'UC": 1 / r2["f_dom_uc"] if r2["f_dom_uc"] else None,
                "Autocorrélation de l'UC": r3["periode_auto_uc_s"],
                "Détection de pics": r1["periode_contr_s"]}
    st.dataframe(pd.DataFrame({"Méthode": list(periodes),
                               "Période (min)": [None if v is None else round(v / 60, 2) for v in periodes.values()]}),
                 width="stretch", hide_index=True)
    vals = [v for v in periodes.values() if v]
    if len(vals) >= 2:
        ecart = (max(vals) - min(vals)) / (sum(vals) / len(vals))
        if ecart <= 0.15:
            st.success(f"Méthodes concordantes (écart {100 * ecart:.0f} %) : période ≈ {sum(vals) / len(vals) / 60:.1f} min.")
        else:
            st.warning(f"Méthodes divergentes (écart {100 * ecart:.0f} %) : contractions irrégulières ou seuils à ajuster.")
    if r3["tau_s"] is not None:
        st.subheader("Relation FHR / UC")
        sens = "baisse" if r3["coef"] < 0 else "monte"
        ordre = "après" if r3["tau_s"] > 0 else "avant"
        st.markdown(f"- Corrélation croisée : coefficient **{r3['coef']:+.2f}**, le FHR {sens} quand l'UC augmente, "
                    f"environ **{abs(r3['tau_s']):.0f} s {ordre}** l'UC.")
        dec = r1["decalages_dec"]
        if dec:
            st.markdown(f"- {len(dec)} décélération(s) associées à une contraction, décalage moyen **{sum(dec) / len(dec):+.0f} s**.")


# --------------------------------------------------------------------------- Onglets
onglets = st.tabs([" Synthèse", " Signal et événements", " Fréquences", " FHR et contractions", " Temps-fréquence"])
for onglet, fonction in zip(onglets, (synthese, P1.afficher, P2.afficher, P3.afficher, P4.afficher)):
    with onglet:
        executer(fonction)

st.caption("Outil d'aide à l'analyse du signal ")