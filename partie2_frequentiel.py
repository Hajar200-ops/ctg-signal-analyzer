"""PARTIE 2 — MEMBRE 2 : analyse fréquentielle (FFT, bandes d'énergie, STFT).

Signaux et Systèmes : transformée de Fourier discrète, fenêtre de Hann, résolution
fréquentielle, compromis temps-fréquence.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import periodogram, stft, welch

from commun import COULEURS, bandeau_partie, layout, notions, plus_long_segment

NOTIONS = r"""
- **Transformée de Fourier discrète** : $X(f) = \sum_n x[n]\, e^{-j 2\pi f n / f_e}$ — elle décompose le signal en fréquences.
- **Fenêtre de Hann** : on multiplie le segment par une cloche pour limiter les « fuites » spectrales.
  On retire d'abord la moyenne (composante continue), ce n'est pas un filtre.
- **Résolution fréquentielle** : $\Delta f = 1/T$ (T = durée du segment). Plus l'enregistrement est long, plus le spectre est précis.
- **Période** d'une composante : $T = 1/f$. La fréquence dominante de l'UC donne le rythme des contractions
  (une contraction toutes les 5 min correspond à $f \approx 0{,}0033$ Hz).
- **Parseval** : l'énergie est la même dans les deux domaines, $\sum_n |x[n]|^2 = \frac{1}{N}\sum_k |X[k]|^2$.
- **Densité spectrale de puissance (DSP)** : répartition de la puissance par Hz. Le périodogramme (une seule FFT) est
  très bruité ; la **méthode de Welch** moyenne plusieurs segments : moins de variance, mais moins de résolution.
- **STFT** : FFT sur des fenêtres glissantes → spectrogramme (temps × fréquence).
  **Compromis** : fenêtre longue = bonne résolution en fréquence mais mauvaise en temps, et inversement.
"""
BANDES = [("Très basses (VLF) < 0,03 Hz", 0.0, 0.03), ("Basses (LF) 0,03–0,15 Hz", 0.03, 0.15),
          ("Moyennes (MF) 0,15–0,5 Hz", 0.15, 0.5), ("Hautes (HF) 0,5–1 Hz", 0.5, 1.0)]


def spectre(signal, fs):
    """Spectre d'amplitude avec fenêtre de Hann. Retourne (fréquences, amplitudes)."""
    fen = np.hanning(len(signal))
    centre = signal - np.average(signal, weights=fen)
    X = np.fft.rfft(centre * fen)
    return np.fft.rfftfreq(len(signal), 1 / fs), 2 * np.abs(X) / np.sum(fen)


def frequence_dominante(f, amp, n):
    """Pic principal en ignorant les 2 premiers « bins » (lobe de la fenêtre de Hann)."""
    m = f >= 2 * (f[1] if len(f) > 1 else 0)
    return float(f[m][np.argmax(amp[m])]) if m.any() else None


def _segment(x, valide):
    seg = plus_long_segment(valide)
    return None if seg is None else (seg[0], seg[1], x[seg[0]:seg[1]])


def afficher(ctx):
    bandeau_partie(2, "Analyse fréquentielle", "Membre 2",
                   "Quelles fréquences contient le signal ? Quel est le rythme des contractions ? "
                   "Comment ces fréquences évoluent-elles dans le temps ?")
    notions("Fourier, fenêtrage, STFT", NOTIONS)
    seg_f = _segment(ctx.fhr, np.isfinite(ctx.fhr))
    if seg_f is None or seg_f[1] - seg_f[0] < 64:
        st.warning("Segment FHR continu trop court (< 64 mesures) pour une analyse fréquentielle.")
        return
    a, b, sig = seg_f
    st.caption(f"Segment FHR continu le plus long : {(b - a) / ctx.fs:.0f} s "
               f"(de {ctx.t[a] / 60:.1f} à {ctx.t[b - 1] / 60:.1f} min) — aucune donnée n'est inventée.")

    st.subheader("FFT du FHR")
    zones = {"0 – 0,1 Hz (lent : contractions, tendances)": 0.1, "0 – 0,5 Hz": 0.5, "Tout (0 – fe/2)": ctx.fs / 2}
    fmax = zones[st.selectbox("Zone affichée", list(zones), key="p2_zone")]
    f, amp = spectre(sig, ctx.fs)
    fd = frequence_dominante(f, amp, len(sig))
    c = st.columns(3)
    c[0].metric("Fréquence dominante", f"{fd:.4f} Hz" if fd else "—")
    c[1].metric("Période correspondante", f"{1 / fd / 60:.1f} min" if fd else "—")
    c[2].metric("Résolution Δf = 1/T", f"{ctx.fs / len(sig):.5f} Hz")
    m = f <= fmax
    fig = go.Figure(go.Scatter(x=f[m], y=amp[m], line=dict(color=COULEURS["fhr"]), name="FHR"))
    if fd and fd <= fmax:
        fig.add_vline(x=fd, line_dash="dot", line_color="black")
    fig.update_xaxes(title_text="Fréquence (Hz)")
    fig.update_yaxes(title_text="Amplitude (BPM)")
    st.plotly_chart(layout(fig, "Spectre d'amplitude du FHR", 360, hover="closest"), width="stretch")

    energie = amp[1:] ** 2
    fr = f[1:]
    tot = energie.sum()
    st.markdown("**Répartition de l'énergie par bande de fréquences**")
    lignes = {nom: f"{100 * energie[(fr >= lo) & (fr < hi)].sum() / tot:.1f} %" for nom, lo, hi in BANDES}
    st.dataframe({"Bande": list(lignes), "Part de l'énergie": list(lignes.values())}, width="stretch", hide_index=True)
    st.caption("Bandes usuelles en variabilité du rythme cardiaque fœtal : à confirmer dans votre référence de cours.")

    st.subheader("FFT de l'UC : rythme des contractions")
    if ctx.uc is None:
        st.info("Pas de signal UC.")
    else:
        seg_u = _segment(ctx.uc, np.isfinite(ctx.uc))
        if seg_u is None or seg_u[1] - seg_u[0] < 64:
            st.warning("Segment UC trop court.")
        else:
            fu, au = spectre(seg_u[2], ctx.fs)
            fdu = frequence_dominante(fu, au, len(seg_u[2]))
            c = st.columns(2)
            c[0].metric("Fréquence dominante UC", f"{fdu:.4f} Hz" if fdu else "—")
            c[1].metric("Période des contractions (1/f)", f"{1 / fdu / 60:.1f} min" if fdu else "—")
            mu = fu <= 0.05
            fig = go.Figure(go.Scatter(x=fu[mu], y=au[mu], line=dict(color=COULEURS["uc"]), name="UC"))
            if fdu and fdu <= 0.05:
                fig.add_vline(x=fdu, line_dash="dot", line_color="black")
            fig.update_xaxes(title_text="Fréquence (Hz)")
            fig.update_yaxes(title_text="Amplitude (UA)")
            st.plotly_chart(layout(fig, "Spectre d'amplitude de l'UC (0 – 0,05 Hz)", 340, hover="closest"), width="stretch")

    st.subheader("Analyse temps-fréquence : STFT du FHR")
    options = [n for n in (64, 128, 256, 512) if n <= len(sig)]
    nper = st.select_slider("Taille de la fenêtre (échantillons)", options=options, value=options[min(1, len(options) - 1)], key="p2_nper")
    fr_s, t_s, Z = stft(sig, fs=ctx.fs, window="hann", nperseg=nper, noverlap=int(nper * 0.75), detrend="constant", boundary=None)
    c = st.columns(3)
    c[0].metric("Durée d'une fenêtre", f"{nper / ctx.fs:.0f} s")
    c[1].metric("Résolution en fréquence", f"{ctx.fs / nper:.3f} Hz")
    c[2].metric("Chevauchement", "75 %")
    fig = go.Figure(go.Heatmap(x=(t_s + ctx.t[a]) / 60, y=fr_s, z=np.abs(Z), colorscale="Viridis", colorbar=dict(title="BPM")))
    fig.update_yaxes(title_text="Fréquence (Hz)", range=[0, 0.5])
    fig.update_xaxes(title_text="Temps (min)")
    st.plotly_chart(layout(fig, "Spectrogramme (STFT)", 420, hover="closest"), width="stretch")
    st.caption("Essayez 64 puis 512 : la fenêtre courte localise mieux dans le temps, la longue sépare mieux les fréquences.")
    _parseval_welch_ui(ctx, sig)


def _parseval_welch_ui(ctx, sig):
    st.subheader("Théorème de Parseval et densité spectrale de puissance")
    x = sig - sig.mean()
    N = len(x)
    E_t = float(np.sum(x ** 2))
    E_f = float(np.sum(np.abs(np.fft.fft(x)) ** 2) / N)
    c = st.columns(3)
    c[0].metric("Énergie dans le temps  Σ|x|²", f"{E_t:,.0f}".replace(",", " "))
    c[1].metric("Énergie en fréquence  (1/N)Σ|X|²", f"{E_f:,.0f}".replace(",", " "))
    c[2].metric("Écart relatif", f"{abs(E_t - E_f) / E_t:.1e}")
    st.caption("Parseval vérifié numériquement : la FFT ne perd ni ne crée d'énergie, elle la redistribue entre les fréquences.")
    options = [n for n in (256, 512, 1024, 2048, 4096) if n <= N // 2]
    if not options:
        st.info("Segment trop court pour la méthode de Welch.")
        return
    nper = st.select_slider("Longueur des segments de Welch (échantillons)", options=options,
                            value=options[min(2, len(options) - 1)], key="p2_welch")
    fw, Pw = welch(x, fs=ctx.fs, window="hann", nperseg=nper, noverlap=nper // 2, detrend="constant")
    fp, Pp = periodogram(x, fs=ctx.fs, window="hann", detrend="constant")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp, y=Pp, name="Périodogramme (1 FFT)", line=dict(color="#CBD5E1")))
    fig.add_trace(go.Scatter(x=fw, y=Pw, name=f"Welch ({nper} éch., 50 % de recouvrement)", line=dict(color=COULEURS["fhr"], width=2)))
    fig.update_xaxes(title_text="Fréquence (Hz)", range=[0, 0.5])
    fig.update_yaxes(title_text="DSP (BPM²/Hz)", type="log")
    st.plotly_chart(layout(fig, "Densité spectrale de puissance du FHR", 360, hover="closest"), width="stretch")
    st.caption(f"Résolution de Welch : {ctx.fs / nper:.4f} Hz (périodogramme : {ctx.fs / N:.5f} Hz). "
               "Compromis : plus de segments = courbe plus lisse, mais pics moins fins.")


def resultats(ctx):
    """Chiffres clés de la partie 2 (sans affichage) : utilisés par l'onglet Synthèse."""
    out = {"f_dom_fhr": None, "f_dom_uc": None}
    seg = _segment(ctx.fhr, np.isfinite(ctx.fhr))
    if seg is not None and seg[1] - seg[0] >= 64:
        f, a = spectre(seg[2], ctx.fs)
        out["f_dom_fhr"] = frequence_dominante(f, a, len(seg[2]))
    if ctx.uc is not None:
        seg = _segment(ctx.uc, np.isfinite(ctx.uc))
        if seg is not None and seg[1] - seg[0] >= 64:
            f, a = spectre(seg[2], ctx.fs)
            out["f_dom_uc"] = frequence_dominante(f, a, len(seg[2]))
    return out
