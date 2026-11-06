"""PARTIE 3 — MEMBRE 3 : autocorrélation et corrélation croisée FHR / UC.

Signaux et Systèmes : autocorrélation (périodicité), corrélation croisée (décalage entre deux signaux),
normalisation, interprétation du signe et du retard.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import coherence, correlate, csd, fftconvolve, find_peaks, welch

from commun import COULEURS, bandeau_partie, interpoler, layout, notions, plus_long_segment

NOTIONS = r"""
- **Autocorrélation** : $R_{xx}(\tau) = \sum_n x[n]\,x[n+\tau]$, normalisée pour valoir 1 en $\tau = 0$.
  Un pic à $\tau_0$ signifie que le signal se « ressemble » après un décalage $\tau_0$ : c'est sa **période**.
- **Corrélation croisée** : $R_{xy}(\tau) = \sum_n x[n]\,y[n+\tau]$ — compare deux signaux décalés.
  Normalisée entre −1 et 1 après retrait des moyennes.
- **Lecture ici** (x = FHR, y = UC) : un décalage **positif** signifie que le FHR réagit **après** l'UC ;
  un coefficient **négatif** signifie que le FHR baisse quand l'UC monte.
- **Attention** : une corrélation forte montre une relation temporelle, pas une cause.
- **Réponse impulsionnelle** : $h(t) = \mathcal{F}^{-1}\{H(f)\}$. Si le système est linéaire et invariant, le FHR est la
  **convolution** de l'UC avec $h$ : $y(t) = (u * h)(t)$. On vérifie le modèle en comparant la sortie prédite à la mesure ($R^2$).
- **Système** : on voit le FHR comme la **sortie** d'un système dont l'**entrée** est l'UC. Sa fonction de transfert
  estimée est $H(f) = P_{xy}(f)/P_{xx}(f)$ (gain $|H|$ et phase), et la **cohérence**
  $C_{xy}(f) = |P_{xy}|^2 / (P_{xx}P_{yy})$ (entre 0 et 1) dit à quel point la relation entrée-sortie est linéaire à chaque fréquence.
"""


def autocorrelation(x, fs):
    """Autocorrélation normalisée. Retourne (décalages en s, valeurs, période en s ou None)."""
    x = x - x.mean()
    n = len(x)
    r = correlate(x, x, mode="full", method="fft")[n - 1:]
    if r[0] != 0:
        r = r / r[0]
    lags = np.arange(n) / fs
    periode = None
    neg = np.where(r < 0)[0]            # on cherche le 1er pic APRÈS le lobe central
    if len(neg):
        pics, _ = find_peaks(r[neg[0]:], prominence=0.1)
        if len(pics):
            periode = float(lags[neg[0] + pics[0]])
    return lags, r, periode


def correlation_croisee(x, y, fs, max_lag_s):
    """Retourne (décalages en s, valeurs, décalage du maximum en valeur absolue, coefficient signé)."""
    x, y = x - x.mean(), y - y.mean()
    norme = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if norme == 0:
        return None
    r = correlate(x, y, mode="full", method="fft") / norme
    lags = np.arange(-(len(x) - 1), len(x)) / fs
    m = np.abs(lags) <= max_lag_s
    lags, r = lags[m], r[m]
    k = int(np.argmax(np.abs(r)))
    return lags, r, float(lags[k]), float(r[k])


def _auto_ui(nom, x, fs, couleur, limite_s):
    lags, r, per = autocorrelation(x, fs)
    c = st.columns(2)
    c[0].metric(f"1er pic de l'autocorrélation ({nom})", f"{per / 60:.1f} min ({per:.0f} s)" if per else "aucun")
    c[1].metric("Durée analysée", f"{len(x) / fs / 60:.1f} min")
    m = lags <= min(limite_s, lags[-1] / 2)
    fig = go.Figure(go.Scatter(x=lags[m] / 60, y=r[m], line=dict(color=couleur), name=nom))
    if per:
        fig.add_vline(x=per / 60, line_dash="dot", line_color="black")
    fig.update_xaxes(title_text="Décalage τ (min)")
    fig.update_yaxes(title_text="Autocorrélation normalisée")
    st.plotly_chart(layout(fig, f"Autocorrélation de {nom}", 330, hover="closest"), width="stretch")


def afficher(ctx):
    bandeau_partie(3, "Corrélations", "Membre 3",
                   "Les contractions sont-elles périodiques ? Le rythme cardiaque réagit-il aux contractions, "
                   "et avec quel retard ?")
    notions("autocorrélation et corrélation croisée", NOTIONS)

    st.subheader("Autocorrélation")
    seg = plus_long_segment(np.isfinite(ctx.fhr))
    if seg is None or seg[1] - seg[0] < 64:
        st.warning("Segment FHR continu trop court.")
    else:
        _auto_ui("FHR", ctx.fhr[seg[0]:seg[1]], ctx.fs, COULEURS["fhr"], 900)
    if ctx.uc is not None:
        su = plus_long_segment(np.isfinite(ctx.uc))
        if su is not None and su[1] - su[0] >= 64:
            _auto_ui("UC", ctx.uc[su[0]:su[1]], ctx.fs, COULEURS["uc"], 1800)
            st.caption("Le premier pic de l'autocorrélation de l'UC donne la période entre deux contractions successives.")

    st.subheader("Corrélation croisée FHR / UC")
    if ctx.uc is None:
        st.info("Sélectionnez une colonne UC pour calculer la corrélation croisée.")
        return
    commun = plus_long_segment(np.isfinite(ctx.fhr) & np.isfinite(ctx.uc))
    if commun is None or commun[1] - commun[0] < 64:
        st.warning("Aucun segment commun valide assez long.")
        return
    a, b = commun
    x, y = ctx.fhr[a:b], ctx.uc[a:b]
    max_lag = st.slider("Décalage maximal exploré (s)", 30, 600, 180, 10, key="p3_lag")
    res = correlation_croisee(x, y, ctx.fs, max_lag)
    if res is None:
        st.warning("Signal constant : corrélation impossible.")
        return
    lags, r, tau, coef = res
    c = st.columns(3)
    c[0].metric("Décalage du maximum |corrélation|", f"{tau:+.0f} s")
    c[1].metric("Coefficient", f"{coef:+.2f}")
    c[2].metric("Segment commun", f"{(b - a) / ctx.fs / 60:.1f} min")
    fig = go.Figure(go.Scatter(x=lags, y=r, line=dict(color=COULEURS["fhr"]), name="Rxy"))
    fig.add_vline(x=0, line_dash="dash", line_color=COULEURS["gris"], annotation_text="décalage nul", annotation_position="top left")
    fig.add_vline(x=tau, line_dash="dot", line_color="black", annotation_text=f"τ = {tau:+.0f} s", annotation_position="top right")
    fig.update_xaxes(title_text="Décalage τ (s) — positif : FHR en retard sur UC")
    fig.update_yaxes(title_text="Corrélation normalisée")
    st.plotly_chart(layout(fig, "Corrélation croisée FHR / UC", 360, hover="closest"), width="stretch")
    force = "faible" if abs(coef) < 0.2 else ("modérée" if abs(coef) < 0.5 else "forte")
    sens = "baisse quand l'UC augmente" if coef < 0 else "monte quand l'UC augmente"
    ordre = "après" if tau > 0 else "avant"
    _systeme_ui(ctx)
    st.info(f"Corrélation **{force}** (coef = {coef:+.2f}) : le FHR {sens}, et il varie environ **{abs(tau):.0f} s {ordre}** l'UC. "
            "Une relation temporelle ne prouve pas une cause, et ce résultat ne constitue pas un diagnostic.")


def _systeme_ui(ctx):
    st.subheader("Le FHR vu comme la sortie d'un système (entrée = UC)")
    comble = st.checkbox("Combler les interruptions du FHR par interpolation linéaire (approximation)", value=False, key="p3_interp")
    if comble:
        fhr_seg, uc_seg = interpoler(ctx.fhr), interpoler(ctx.uc)
    else:
        seg = plus_long_segment(np.isfinite(ctx.fhr) & np.isfinite(ctx.uc))
        if seg is None:
            st.info("Aucun segment commun valide.")
            return
        fhr_seg, uc_seg = ctx.fhr[seg[0]:seg[1]], ctx.uc[seg[0]:seg[1]]
    # fréquence des contractions : pic du spectre de l'UC sur son plus long segment (meilleure résolution)
    su = plus_long_segment(np.isfinite(ctx.uc))
    u_long = ctx.uc[su[0]:su[1]]
    fen = np.hanning(len(u_long))
    F = np.fft.rfftfreq(len(u_long), 1 / ctx.fs)
    A = np.abs(np.fft.rfft((u_long - np.average(u_long, weights=fen)) * fen))
    bande = (F >= 0.002) & (F <= 0.05)
    f_contr = F[bande][int(np.argmax(A[bande]))]
    # contraintes : ≥ 2 périodes de contraction par segment (résolution), ≥ 5 segments moyennés (fiabilité)
    n_min, n_max = int(np.ceil(2 * ctx.fs / f_contr)), len(uc_seg) // 3
    if n_min > n_max:
        st.info(f"Segment trop court pour cette analyse : à la fréquence des contractions ({f_contr:.4f} Hz, période "
                f"{1 / f_contr / 60:.1f} min), il faut au moins {3 * n_min / ctx.fs / 60:.0f} min continues "
                f"(ici {len(uc_seg) / ctx.fs / 60:.0f} min). Essayez de combler les interruptions.")
        return
    nper = st.slider("Longueur des segments de Welch (échantillons)", n_min, n_max, min(n_max, max(n_min, 2 * n_min)), 16, key="p3_coh")
    kw = dict(fs=ctx.fs, window="hann", nperseg=nper, noverlap=nper // 2)
    f, Cxy = coherence(uc_seg, fhr_seg, **kw)
    _, Pxy = csd(uc_seg, fhr_seg, **kw)
    _, Pxx = welch(uc_seg, **kw)
    H = Pxy / np.maximum(Pxx, 1e-12)
    k = int(np.argmin(np.abs(f - f_contr)))
    c = st.columns(4)
    c[0].metric("Fréquence des contractions", f"{f_contr:.4f} Hz")
    c[1].metric("Période", f"{1 / f_contr / 60:.1f} min")
    c[2].metric("Cohérence à cette fréquence", f"{Cxy[k]:.2f}")
    c[3].metric("Gain |H| à cette fréquence", f"{abs(H[k]):.2f} BPM/UA")
    m = f <= 0.05
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Cohérence UC → FHR", "Gain |H(f)| (BPM par UA)", "Phase de H(f) (degrés)"))
    fig.add_trace(go.Scatter(x=f[m], y=Cxy[m], line=dict(color=COULEURS["fhr"]), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=f[m], y=np.abs(H[m]), line=dict(color=COULEURS["uc"]), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=f[m], y=np.degrees(np.angle(H[m])), line=dict(color=COULEURS["gris"]), showlegend=False), 3, 1)
    for r in (1, 2, 3):
        fig.add_vline(x=f[k], line_dash="dot", line_color="black", row=r, col=1)
    fig.update_xaxes(title_text="Fréquence (Hz)", row=3, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    st.plotly_chart(layout(fig, "", 640, hover="x unified"), width="stretch")

    # --- réponse impulsionnelle h(t) = TF⁻¹{H(f)·Cxy(f)} et modèle par convolution : FHR ≈ moyenne + (UC * h)
    # (H est pondérée par la cohérence : là où l'UC n'explique pas le FHR, on réduit l'influence de l'estimation)
    h = np.fft.fftshift(np.fft.irfft(H * Cxy, n=nper))
    lags_h = (np.arange(nper) - nper // 2) / ctx.fs
    u0, y0 = uc_seg - uc_seg.mean(), fhr_seg - fhr_seg.mean()
    y_pred = fftconvolve(u0, h, mode="same")
    r2 = 1 - np.var(y0 - y_pred) / np.var(y0)
    c = st.columns(2)
    c[0].metric("Part de variance du FHR expliquée (R²)", f"{max(r2, 0):.2f}")
    c[1].metric("Longueur de h", f"{nper / ctx.fs:.0f} s")
    fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                         subplot_titles=("Réponse impulsionnelle estimée h(t) (BPM par UA)", "FHR mesuré et FHR prédit par convolution UC * h"))
    mh = (lags_h >= -120) & (lags_h <= 240)
    fig2.add_trace(go.Scatter(x=lags_h[mh], y=h[mh], line=dict(color=COULEURS["uc"]), showlegend=False), 1, 1)
    fig2.add_vline(x=0, line_dash="dash", line_color=COULEURS["gris"], row=1, col=1)
    tt = np.arange(len(y0)) / ctx.fs / 60
    pas = max(1, len(tt) // 1500)                                   # sous-échantillonnage pour l'AFFICHAGE seulement
    fig2.add_trace(go.Scatter(x=tt[::pas], y=y0[::pas], name="FHR mesuré", line=dict(color=COULEURS["fhr"], width=1)), 2, 1)
    fig2.add_trace(go.Scatter(x=tt[::pas], y=y_pred[::pas], name="FHR prédit", line=dict(color=COULEURS["uc"], width=2)), 2, 1)
    fig2.update_xaxes(title_text="Décalage (s)", row=1, col=1)
    fig2.update_xaxes(title_text="Temps (min)", row=2, col=1)
    fig2.update_yaxes(title_text="BPM (centré)", row=2, col=1)
    st.plotly_chart(layout(fig2, "", 560, hover="x unified"), width="stretch")
    st.caption(f"Le modèle linéaire explique {max(r2, 0):.0%} de la variance du FHR ; le reste vient de ce que l'UC ne prédit pas "
               "(accélérations spontanées, bruit, non-linéarités). R² est calculé sur les données qui ont servi à estimer h : "
               "il est optimiste. Attention : quand l'UC est presque périodique, seules ses harmoniques « excitent » le système "
               "et h est mal déterminée (ne lisez pas un retard précis sur h : utilisez la corrélation croisée pour le retard).")
    forte = Cxy[k] >= 0.7
    st.info(f"À la fréquence des contractions, la cohérence vaut **{Cxy[k]:.2f}** : "
            + ("la relation entre UC et FHR est **fortement linéaire** à cette fréquence." if forte else
               "la relation linéaire entre UC et FHR est **limitée** à cette fréquence (le FHR varie aussi pour d'autres raisons).")
            + f" Une phase de {np.degrees(np.angle(H[k])):+.0f}° proche de ±180° indique que le FHR varie en **sens inverse** de l'UC. "
              "Chaque segment contient au moins 2 périodes de contraction et au moins 5 segments sont moyennés ; "
              "une cohérence ne prouve pas une cause.")


def resultats(ctx):
    """Chiffres clés de la partie 3 (sans affichage) : utilisés par l'onglet Synthèse."""
    out = {"periode_auto_uc_s": None, "tau_s": None, "coef": None}
    if ctx.uc is None:
        return out
    su = plus_long_segment(np.isfinite(ctx.uc))
    if su is not None and su[1] - su[0] >= 64:
        out["periode_auto_uc_s"] = autocorrelation(ctx.uc[su[0]:su[1]], ctx.fs)[2]
    cm = plus_long_segment(np.isfinite(ctx.fhr) & np.isfinite(ctx.uc))
    if cm is not None and cm[1] - cm[0] >= 64:
        res = correlation_croisee(ctx.fhr[cm[0]:cm[1]], ctx.uc[cm[0]:cm[1]], ctx.fs, 180)
        if res is not None:
            out["tau_s"], out["coef"] = res[2], res[3]
    return out
