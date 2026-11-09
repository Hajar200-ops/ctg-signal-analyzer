"""Signal et événements : informations, qualité, interruptions, détection d'événements, indicateurs usuels."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from commun import COULEURS, bandeau_partie, interpoler, layout


# ---------------------------------------------------------------------------
# Détection d'événements
# ---------------------------------------------------------------------------
def _franchissement(x, p, niveau, sens, k):
    """Indice où x passe sous `niveau` (k échantillons consécutifs) en partant du sommet p."""
    n, i, run = len(x), p, 0
    while 0 <= i + sens < n:
        i += sens
        run = run + 1 if x[i] < niveau else 0
        if run >= k:
            return i - sens * (k - 1)
    return i


def _evenements(t, x, pics, fs, duree_min_s, interdit):
    """x orienté pour que l'événement soit un maximum. Début/fin/durée = largeur à mi-hauteur."""
    k, evs = max(1, int(round(2 * fs))), []
    for p in pics:
        niveau = x[p] / 2.0
        g, d = _franchissement(x, p, niveau, -1, k), _franchissement(x, p, niveau, +1, k)
        if (d - g) / fs < duree_min_s or interdit[g:d + 1].any():
            continue
        evs.append({"i": int(p), "t": float(t[p]), "debut": float(t[g]), "fin": float(t[d]),
                    "milieu": float((t[g] + t[d]) / 2), "duree": float((d - g) / fs)})
    return evs


def detecter(ctx, prm):
    """Retourne (baseline_fhr, baseline_uc, accélérations, décélérations, contractions)."""
    t, fs = ctx.t, ctx.fs
    invalide = ~np.isfinite(ctx.fhr)
    base = float(np.nanmedian(ctx.fhr))
    # on n'accepte pas d'événement qui chevauche une interruption longue (signal inventé par interpolation)
    longs = np.zeros(len(t), bool)
    d = np.diff(np.r_[False, invalide, False].astype(int))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if (b - a) / fs > prm["gap_max_s"]:
            longs[a:b] = True
    x = interpoler(ctx.fhr) - base
    kw = dict(prominence=prm["seuil"], distance=max(1, int(prm["dist_fhr_s"] * fs)))
    pa, _ = find_peaks(x, height=prm["seuil"], **kw)
    pc, _ = find_peaks(-x, height=prm["seuil"], **kw)
    acc = _evenements(t, x, pa, fs, prm["duree_min"], longs)
    dec = _evenements(t, -x, pc, fs, prm["duree_min"], longs)
    for e in acc + dec:
        e["amplitude"] = float(x[e["i"]])          # signé : + accélération, − décélération
    contr, base_uc = [], None
    if ctx.uc is not None and np.isfinite(ctx.uc).sum() > 10:
        u = interpoler(ctx.uc)
        base_uc = float(np.median(u))
        dyn = float(np.percentile(u, 99) - base_uc)
        if dyn > 0:
            pk, _ = find_peaks(u - base_uc, height=prm["seuil_uc"] * dyn, prominence=prm["seuil_uc"] * dyn,
                               distance=max(1, int(prm["dist_uc_s"] * fs)))
            contr = _evenements(t, u - base_uc, pk, fs, prm["duree_min_uc"], np.zeros(len(t), bool))
            for e in contr:
                e["amplitude"] = float(u[e["i"]] - base_uc)
    for e in acc + dec:                              # décalage par rapport à la contraction la plus proche
        e["decalage"] = None
        if contr:
            j = int(np.argmin([abs(e["milieu"] - c["milieu"]) for c in contr]))
            dlt = e["milieu"] - contr[j]["milieu"]
            if abs(dlt) <= prm["fenetre_assoc_s"]:
                e["decalage"] = float(dlt)
                e["contraction"] = j
    return base, base_uc, acc, dec, contr


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------
def afficher(ctx):
    bandeau_partie("🍼", "Signal et événements")
    _informations(ctx)
    _statistiques(ctx)
    _visualisation(ctx)
    _qualite(ctx)
    _interruptions(ctx)
    res = _evenements_ui(ctx)
    if res:
        _indicateurs_ui(ctx, *res)


def _informations(ctx):
    st.subheader("Informations sur l'enregistrement")
    c = st.columns(5)
    c[0].metric("Mesures", f"{len(ctx.t):,}".replace(",", " "))
    c[1].metric("Durée", f"{ctx.duree_s / 60:.1f} min")
    c[2].metric("Fréquence d'échantillonnage", f"{ctx.fs:.2f} Hz")
    c[3].metric("Nyquist (fe/2)", f"{ctx.fs / 2:.2f} Hz")
    c[4].metric("Valeurs manquantes (fichier)", int(ctx.df.isna().sum().sum()))
    with st.expander("Aperçu du fichier"):
        st.dataframe(ctx.df.head(30), width="stretch")


def _statistiques(ctx):
    st.subheader("Statistiques du FHR (valeurs valides)")
    v = ctx.fhr[np.isfinite(ctx.fhr)]
    c = st.columns(5)
    c[0].metric("Moyenne", f"{v.mean():.1f} BPM")
    c[1].metric("Minimum", f"{v.min():.1f} BPM")
    c[2].metric("Maximum", f"{v.max():.1f} BPM")
    c[3].metric("Variance", f"{v.var():.1f}")
    c[4].metric("Écart-type", f"{v.std():.1f}")


def _iqr_aberrantes(fhr):
    v = fhr[np.isfinite(fhr)]
    q1, q3 = np.percentile(v, [25, 75])
    bas, haut = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return np.isfinite(fhr) & ((fhr < bas) | (fhr > haut)), bas, haut


def _visualisation(ctx):
    st.subheader("Visualisation des signaux")
    tm = ctx.t / 60
    deux = ctx.uc is not None
    fig = make_subplots(rows=2 if deux else 1, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Rythme cardiaque fœtal (FHR)", "Contractions utérines (UC)")[:2 if deux else 1])
    fig.add_trace(go.Scattergl(x=tm, y=ctx.fhr, name="FHR", line=dict(color=COULEURS["fhr"], width=1)), 1, 1)
    ab, _, _ = _iqr_aberrantes(ctx.fhr)
    if ab.any():
        fig.add_trace(go.Scatter(x=tm[ab], y=ctx.fhr[ab], mode="markers", name="Valeurs à vérifier (IQR)",
                                 marker=dict(color="black", size=6, symbol="x")), 1, 1)
    if deux:
        fig.add_trace(go.Scattergl(x=tm, y=ctx.uc, name="UC", line=dict(color=COULEURS["uc"], width=1)), 2, 1)
    fig.update_xaxes(title_text="Temps (min)", row=2 if deux else 1, col=1)
    fig.update_yaxes(title_text="BPM", row=1, col=1)
    if deux:
        fig.update_yaxes(title_text="UA", row=2, col=1)
    st.plotly_chart(layout(fig, hauteur=560 if deux else 340), width="stretch")


def _qualite(ctx):
    st.subheader("Contrôle de qualité")
    brut, n = ctx.fhr_brut, len(ctx.t)
    valides = int(np.isfinite(ctx.fhr).sum())
    _, bas, haut = _iqr_aberrantes(ctx.fhr)
    c = st.columns(4)
    c[0].metric("FHR valides", f"{100 * valides / n:.1f} %")
    c[1].metric("FHR = 0", int((brut == 0).sum()))
    c[2].metric("FHR manquants", int(np.isnan(brut).sum()))
    c[3].metric("Statistiquement inhabituels", int(_iqr_aberrantes(ctx.fhr)[0].sum()))
    st.caption(f"Valeurs hors de l'intervalle {bas:.0f} – {haut:.0f} BPM (méthode IQR).")
    if ctx.uc is None:
        st.info("Pas de signal UC : contrôle de qualité UC indisponible.")
        return
    u = ctx.uc
    c = st.columns(4)
    c[0].metric("UC valides", f"{100 * np.isfinite(u).mean():.1f} %")
    c[1].metric("UC manquants", int(np.isnan(u).sum()))
    c[2].metric("UC = 0", int((u == 0).sum()))
    c[3].metric("UC négatifs", int((u < 0).sum()))


def _interruptions(ctx):
    st.subheader("Interruptions du signal FHR")
    invalide = ~np.isfinite(ctx.fhr)
    d = np.diff(np.r_[False, invalide, False].astype(int))
    debuts, fins = np.where(d == 1)[0], np.where(d == -1)[0]
    dispo = 100 * (1 - invalide.mean())
    c = st.columns(3)
    c[0].metric("Nombre d'interruptions", len(debuts))
    c[1].metric("Durée totale", f"{invalide.sum() / ctx.fs:.1f} s")
    c[2].metric("Disponibilité du signal", f"{dispo:.1f} %")
    if len(debuts) == 0:
        st.success("Aucune interruption détectée.")
        return
    tab = pd.DataFrame({"Début (s)": ctx.t[debuts], "Fin (s)": ctx.t[fins - 1],
                        "Échantillons": fins - debuts, "Durée (s)": (fins - debuts) / ctx.fs})
    st.dataframe(tab.round(2), width="stretch")
    fig = go.Figure(go.Scattergl(x=ctx.t / 60, y=ctx.fhr, name="FHR", line=dict(color=COULEURS["fhr"], width=1)))
    for a, b in list(zip(debuts, fins))[:300]:
        fig.add_vrect(x0=ctx.t[a] / 60, x1=ctx.t[b - 1] / 60 + 1 / ctx.fs / 60, fillcolor="black", opacity=0.15, line_width=0)
    fig.update_xaxes(title_text="Temps (min)")
    fig.update_yaxes(title_text="FHR (BPM)")
    st.plotly_chart(layout(fig, "Localisation des interruptions (zones grisées)", 340), width="stretch")


def _evenements_ui(ctx):
    st.subheader("Détection des accélérations, décélérations et contractions")
    with st.expander("Seuils de détection"):
        c = st.columns(3)
        prm = {
            "seuil": c[0].slider("Seuil FHR (BPM au-dessus/dessous de la base)", 5.0, 40.0, 15.0, 1.0, key="p1_seuil"),
            "duree_min": c[1].slider("Durée min. FHR à mi-hauteur (s)", 5.0, 60.0, 10.0, 1.0, key="p1_dur"),
            "dist_fhr_s": c[2].slider("Distance min. entre événements FHR (s)", 5.0, 120.0, 20.0, 5.0, key="p1_dist"),
            "seuil_uc": c[0].slider("Seuil contraction (fraction de l'amplitude UC)", 0.1, 0.8, 0.3, 0.05, key="p1_suc"),
            "dist_uc_s": c[1].slider("Distance min. entre contractions (s)", 30.0, 300.0, 90.0, 10.0, key="p1_duc"),
            "duree_min_uc": c[2].slider("Durée min. contraction (s)", 10.0, 120.0, 30.0, 5.0, key="p1_dmuc"),
            "gap_max_s": 15.0, "fenetre_assoc_s": 120.0}
    if np.isfinite(ctx.fhr).sum() < 100:
        st.info("Pas assez de données FHR valides.")
        return None
    base, base_uc, acc, dec, contr = detect_cached(ctx, prm)
    c = st.columns(5)
    c[0].metric("Ligne de base FHR", f"{base:.0f} BPM")
    c[1].metric("Accélérations", len(acc))
    c[2].metric("Décélérations", len(dec))
    c[3].metric("Contractions", len(contr) if ctx.uc is not None else "—")
    c[4].metric("Contractions / 10 min", f"{len(contr) * 600 / ctx.duree_s:.1f}" if ctx.uc is not None else "—")

    lignes = []
    for nom, evs in (("Accélération", acc), ("Décélération", dec), ("Contraction", contr)):
        for e in evs:
            lignes.append({"Type": nom, "Instant (min)": e["t"] / 60, "Début (min)": e["debut"] / 60,
                           "Fin (min)": e["fin"] / 60, "Durée (s)": e["duree"], "Amplitude": e["amplitude"],
                           "Décalage vs contraction (s)": e.get("decalage")})
    tab = pd.DataFrame(lignes)
    if not tab.empty:
        tab = tab.sort_values("Instant (min)").reset_index(drop=True)
        st.dataframe(tab.round(2), width="stretch")
        st.download_button("⬇️ Événements (CSV)", tab.round(3).to_csv(index=False).encode("utf-8"), "evenements_ctg.csv", "text/csv")
    else:
        st.info("Aucun événement avec ces seuils.")

    tm = ctx.t / 60
    deux = ctx.uc is not None
    fig = make_subplots(rows=2 if deux else 1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scattergl(x=tm, y=ctx.fhr, name="FHR", line=dict(color=COULEURS["fhr"], width=1)), 1, 1)
    fig.add_hline(y=base, line_dash="dash", line_color=COULEURS["gris"], row=1, col=1)
    for evs, col in ((acc, COULEURS["accel"]), (dec, COULEURS["decel"])):
        for e in evs[:300]:
            fig.add_vrect(x0=e["debut"] / 60, x1=e["fin"] / 60, fillcolor=col, opacity=0.25, line_width=0, row=1, col=1)
    if deux:
        fig.add_trace(go.Scattergl(x=tm, y=ctx.uc, name="UC", line=dict(color=COULEURS["uc"], width=1)), 2, 1)
        if contr:
            fig.add_trace(go.Scatter(x=[e["t"] / 60 for e in contr], y=[ctx.uc[e["i"]] for e in contr], mode="markers",
                                     name="Contraction", marker=dict(symbol="triangle-down", size=10, color="black")), 2, 1)
    fig.update_xaxes(title_text="Temps (min)", row=2 if deux else 1, col=1)
    st.plotly_chart(layout(fig, "Courbes annotées : vert = accélération, orange = décélération", 560 if deux else 340), width="stretch")
    return base, acc, dec, contr


@st.cache_data(show_spinner=False)
def _detect_cache(t, fhr, uc, fs, prm_items):
    from types import SimpleNamespace
    return detecter(SimpleNamespace(t=t, fhr=fhr, uc=uc, fs=fs), dict(prm_items))


def detect_cached(ctx, prm):
    return _detect_cache(ctx.t, ctx.fhr, ctx.uc, ctx.fs, tuple(sorted(prm.items())))


# ---------------------------------------------------------------------------
# Indicateurs usuels d'un CTG, mesurés par traitement du signal puis comparés à des repères
# ---------------------------------------------------------------------------
def _variabilite(ctx, acc, dec):
    """Variabilité = amplitude pic-à-pic du FHR sur 1 min, hors accélérations/décélérations (médiane des fenêtres)."""
    w, marge = int(60 * ctx.fs), int(30 * ctx.fs)
    occupe = np.zeros(len(ctx.t), bool)
    for e in acc + dec:
        i0, i1 = np.searchsorted(ctx.t, e["debut"]), np.searchsorted(ctx.t, e["fin"])
        occupe[max(0, i0 - marge): i1 + marge + 1] = True
    amps = []
    for d in range(0, len(ctx.fhr) - w + 1, w):
        seg = ctx.fhr[d:d + w]
        if np.isfinite(seg).mean() >= 0.8 and not occupe[d:d + w].any():
            amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
    return (float(np.median(amps)), len(amps)) if amps else (None, 0)


def indicateurs(ctx, base, acc, dec, contr):
    """Retourne une liste de lignes (Indicateur, Mesure, Repère, Position). Repères : FIGO 2015."""
    lignes = []
    ok = lambda b: " dans le repère" if b else " hors repère"
    lignes.append(("Ligne de base du FHR", f"{base:.0f} BPM", "110 – 160 BPM", ok(110 <= base <= 160)))
    var, n = _variabilite(ctx, acc, dec)
    lignes.append(("Variabilité (amplitude pic-à-pic, 1 min)", "non évaluable" if var is None else f"{var:.1f} BPM",
                   "5 – 25 BPM", "➖ non évaluable" if var is None else ok(5 <= var <= 25)))
    duree_min = ctx.duree_s / 60
    lignes.append(("Accélérations", f"{len(acc)} en {duree_min:.0f} min", "présentes = rassurant (non obligatoire)",
                   " présentes" if acc else "➖ absentes"))
    if ctx.uc is None:
        lignes.append(("Décélérations", f"{len(dec)}", "pas de décélérations tardives répétitives", "➖ UC absente : timing non évaluable"))
        return lignes
    tardives = [e for e in dec if e.get("decalage") is not None and e["decalage"] >= 10]
    liees = {e["contraction"] for e in tardives}
    part = len(liees) / len(contr) if contr else 0
    repetitives = len(contr) >= 3 and part > 0.5
    lignes.append(("Décélérations", f"{len(dec)} dont {len(tardives)} retardées (≥ 10 s après la contraction)",
                   "pas de décélérations tardives répétitives (> 50 % des contractions)",
                   (f" retardées après {100 * part:.0f} % des contractions" if repetitives else " pas de répétition")
                   if contr else "➖ non évaluable"))
    freq = len(contr) * 600 / ctx.duree_s
    lignes.append(("Fréquence des contractions", f"{freq:.1f} / 10 min", "≤ 5 / 10 min", ok(freq <= 5)))
    return lignes


def _indicateurs_ui(ctx, base, acc, dec, contr):
    st.subheader("Indicateurs usuels d'un CTG et repères")
    lignes = indicateurs(ctx, base, acc, dec, contr)
    st.dataframe(pd.DataFrame(lignes, columns=["Indicateur", "Mesure", "Repère usuel", "Position"]),
                 width="stretch", hide_index=True)


def resultats(ctx):
    """Chiffres clés (sans affichage) : utilisés par l'onglet Synthèse."""
    prm = {"seuil": 15.0, "duree_min": 10.0, "dist_fhr_s": 20.0, "seuil_uc": 0.3, "dist_uc_s": 90.0,
           "duree_min_uc": 30.0, "gap_max_s": 15.0, "fenetre_assoc_s": 120.0}
    base, _, acc, dec, contr = detect_cached(ctx, prm)
    ints = [b["t"] - a["t"] for a, b in zip(contr, contr[1:])]
    return {"indicateurs": indicateurs(ctx, base, acc, dec, contr), "disponibilite": 100 * float(np.isfinite(ctx.fhr).mean()),
            "baseline": base, "n_acc": len(acc), "n_dec": len(dec), "n_contr": len(contr),
            "periode_contr_s": float(np.mean(ints)) if ints else None,
            "decalages_dec": [e["decalage"] for e in dec if e["decalage"] is not None]}