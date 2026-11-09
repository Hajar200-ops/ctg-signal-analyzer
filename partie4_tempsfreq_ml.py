"""Temps-fréquence (ondelettes), caractéristiques par fenêtres, fenêtres inhabituelles."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commun import COULEURS, bandeau_partie, layout, plus_long_segment


@st.cache_data(show_spinner=False)
def _cwt(signal, fs, fmin, fmax, n_freq):
    import pywt
    freqs = np.geomspace(fmin, fmax, n_freq)
    scales = pywt.central_frequency("morl") * fs / freqs
    coefs, f = pywt.cwt(signal, scales, "morl", sampling_period=1 / fs, method="fft")
    return np.abs(coefs), f


def fenetres(t, x, fs, duree_s, valide, nom, unite=""):
    """Caractéristiques par fenêtres consécutives de `duree_s` secondes (≥ 80 % de valeurs valides)."""
    taille = int(duree_s * fs)
    lignes = []
    for k, debut in enumerate(range(0, len(x) - taille + 1, taille)):
        w = x[debut:debut + taille]
        v = w[valide(w)]
        dispo = 100 * len(v) / taille
        if dispo < 80:
            continue
        lignes.append({"Fenêtre": k, "Début (s)": float(t[debut]), "Fin (s)": float(t[debut + taille - 1] + 1 / fs),
                       f"Moyenne {nom}{unite}": v.mean(), f"Minimum {nom}{unite}": v.min(), f"Maximum {nom}{unite}": v.max(),
                       f"Variance {nom}": v.var(), f"Écart-type {nom}": v.std(),
                       f"Énergie centrée {nom}": float(np.sum((v - v.mean()) ** 2)), f"Disponibilité {nom} (%)": dispo})
    return pd.DataFrame(lignes)


def afficher(ctx):
    bandeau_partie("🧸", "Temps-fréquence et fenêtres inhabituelles")
    _cwt_ui(ctx)
    fhr_df, uc_df, comb = _caracteristiques_ui(ctx)
    _anomalies_ui(ctx, comb)


def _cwt_ui(ctx):
    st.subheader("Analyse par ondelettes (CWT, ondelette de Morlet)")
    seg = plus_long_segment(np.isfinite(ctx.fhr))
    if seg is None or seg[1] - seg[0] < 128:
        st.warning("Segment FHR continu trop court pour la CWT.")
        return
    a, b = seg
    sig = ctx.fhr[a:b]
    c = st.columns(2)
    fmin = c[0].select_slider("Fréquence minimale (Hz)", [0.005, 0.01, 0.02, 0.05], value=0.01, key="p4_fmin")
    fmax = c[1].select_slider("Fréquence maximale (Hz)", [0.25, 0.5, 1.0, 1.5], value=1.0, key="p4_fmax")
    if fmax > ctx.fs / 2:
        fmax = ctx.fs / 2
    with st.spinner("Calcul du scalogramme…"):
        mag, f = _cwt(sig - sig.mean(), ctx.fs, fmin, fmax, 60)
    pas = max(1, mag.shape[1] // 1200)          # sous-échantillonnage pour l'AFFICHAGE seulement
    tt = (ctx.t[a:b] / 60)[::pas]
    z = mag[:, ::pas]
    fig = go.Figure(go.Heatmap(x=tt, y=f, z=z, colorscale="Viridis", zmin=0, zmax=float(np.percentile(z, 99)),
                               colorbar=dict(title="Amplitude")))
    reperes = [v for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5) if f.min() * 0.99 <= v <= f.max() * 1.01]
    fig.update_yaxes(title_text="Fréquence (Hz)", type="log", tickvals=reperes, ticktext=[f"{v:g}" for v in reperes])
    fig.update_xaxes(title_text="Temps (min)")
    st.plotly_chart(layout(fig, "Scalogramme du FHR (CWT)", 420, hover="closest"), width="stretch")


def _caracteristiques_ui(ctx):
    st.subheader("Caractéristiques par fenêtres")
    duree = st.slider("Durée d'une fenêtre (s)", 10, 120, 30, 10, key="p4_win")
    fhr_df = fenetres(ctx.t, ctx.fhr, ctx.fs, duree, lambda w: np.isfinite(w) & (w > 0), "FHR", " (BPM)")
    uc_df = fenetres(ctx.t, ctx.uc, ctx.fs, duree, lambda w: np.isfinite(w) & (w >= 0), "UC") if ctx.uc is not None else None
    c = st.columns(3)
    c[0].metric("Échantillons / fenêtre", int(duree * ctx.fs))
    c[1].metric("Fenêtres FHR retenues", len(fhr_df))
    c[2].metric("Fenêtres UC retenues", len(uc_df) if uc_df is not None else "—")
    if fhr_df.empty:
        st.warning("Aucune fenêtre ne possède 80 % de données FHR valides.")
        return fhr_df, uc_df, None
    with st.expander("Tableau FHR"):
        st.dataframe(fhr_df.round(3), width="stretch")
    if uc_df is not None and not uc_df.empty:
        with st.expander("Tableau UC"):
            st.dataframe(uc_df.round(3), width="stretch")
    comb = None
    if uc_df is not None and not uc_df.empty:
        comb = fhr_df.merge(uc_df.drop(columns=["Début (s)", "Fin (s)"]), on="Fenêtre", how="inner")
        st.markdown(f"**Fusion FHR + UC : {len(comb)} fenêtres communes**")
        st.dataframe(comb.round(3), width="stretch")
        st.download_button("⬇️ Caractéristiques FHR+UC (CSV)", comb.to_csv(index=False).encode("utf-8"),
                           "ctg_features_fhr_uc.csv", "text/csv")
    else:
        st.download_button("⬇️ Caractéristiques FHR (CSV)", fhr_df.to_csv(index=False).encode("utf-8"),
                           "ctg_features_fhr.csv", "text/csv")
    mil = (fhr_df["Début (s)"] + fhr_df["Fin (s)"]) / 2 / 60
    fig = go.Figure(go.Scatter(x=mil, y=fhr_df["Moyenne FHR (BPM)"], mode="lines+markers",
                               line=dict(color=COULEURS["fhr"]), name="Moyenne FHR"))
    fig.update_xaxes(title_text="Temps (min)")
    fig.update_yaxes(title_text="BPM")
    st.plotly_chart(layout(fig, f"Moyenne du FHR par fenêtre de {duree} s", 320), width="stretch")
    return fhr_df, uc_df, comb


def _anomalies_ui(ctx, comb):
    st.subheader("Fenêtres inhabituelles (Isolation Forest)")
    if comb is None:
        st.info("La détection utilise les caractéristiques FHR **et** UC : un signal UC est nécessaire.")
        return
    colonnes = [c for c in comb.columns if c not in ("Fenêtre", "Début (s)", "Fin (s)") and not c.startswith("Disponibilité")]
    data = comb[colonnes].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        st.warning("Il faut au moins 10 fenêtres valides.")
        return
    contamination = st.slider("Proportion de fenêtres signalées (%)", 1, 15, 5, key="p4_cont") / 100
    from sklearn.ensemble import IsolationForest
    pred = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit_predict(data)
    res = comb.loc[data.index].copy()
    res["Résultat"] = np.where(pred == -1, "Caractéristiques inhabituelles", "Habituelle")
    c = st.columns(3)
    c[0].metric("Fenêtres habituelles", int((pred == 1).sum()))
    c[1].metric("Fenêtres inhabituelles", int((pred == -1).sum()))
    c[2].metric("Caractéristiques utilisées", len(colonnes))
    st.caption(f"Repérage statistique : {contamination:.0%} des fenêtres sont signalées par construction. Ce n'est pas un diagnostic.")
    fig = go.Figure(go.Scattergl(x=ctx.t / 60, y=ctx.fhr, name="FHR", line=dict(color="#CBD5E1", width=1)))
    for _, r in res[res["Résultat"] != "Habituelle"].iterrows():
        fig.add_vrect(x0=r["Début (s)"] / 60, x1=r["Fin (s)"] / 60, fillcolor=COULEURS["fhr"], opacity=0.3, line_width=0)
    fig.update_xaxes(title_text="Temps (min)")
    fig.update_yaxes(title_text="FHR (BPM)")
    st.plotly_chart(layout(fig, "Fenêtres signalées (zones roses) sur le FHR", 360), width="stretch")
    st.dataframe(res[res["Résultat"] != "Habituelle"][["Début (s)", "Fin (s)", "Moyenne FHR (BPM)", "Moyenne UC", "Résultat"]].round(2),
                 width="stretch")
    st.download_button("⬇️ Résultats (CSV)", res.to_csv(index=False).encode("utf-8"), "ctg_anomalies.csv", "text/csv")