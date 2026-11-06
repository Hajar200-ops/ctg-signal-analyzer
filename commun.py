"""OUTILS PARTAGÉS : lecture de n'importe quel fichier, contexte de données, thème visuel.

Ici il n'y a AUCUN traitement du signal : tout le Signaux et Systèmes est dans les 4 parties.
Chaque partie reçoit le même objet `Contexte` et affiche ses propres résultats.
"""
import os
import re
import tempfile
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Palette (thème « bébé » : rose tendre, bleu ciel, menthe, pêche, lavande)
# ----------------------------------------------------------------------------
COULEURS = {"fhr": "#F43F5E", "uc": "#3B82F6", "accel": "#34D399", "decel": "#FB923C",
            "gris": "#94A3B8", "marine": "#4A2C5A"}
FS_DEFAUT = 4.0   # Hz : fréquence d'échantillonnage des CTG de la base CTU-UHB


# ----------------------------------------------------------------------------
# Contexte : ce que reçoit chaque partie
# ----------------------------------------------------------------------------
@dataclass
class Contexte:
    df: pd.DataFrame
    nom: str
    t: np.ndarray                  # temps en secondes
    fs: float                      # fréquence d'échantillonnage (Hz)
    fhr_brut: np.ndarray           # FHR tel quel dans le fichier (NaN si non numérique)
    fhr: np.ndarray                # FHR avec NaN aux endroits invalides (<= 0 ou manquants)
    uc: "np.ndarray | None"        # UC (None si pas de colonne UC)
    temps_reel: bool               # True si une vraie colonne temps a été fournie
    avertissements: list = field(default_factory=list)

    @property
    def duree_s(self):
        return len(self.t) / self.fs


def construire_contexte(df, col_temps, col_fhr, col_uc, nom="enregistrement",
                        unite_temps="auto", fs_defaut=FS_DEFAUT):
    """Transforme le tableau choisi par l'utilisateur en Contexte. Lève ValueError si inutilisable."""
    av = []
    if col_fhr is None or col_fhr not in df.columns:
        raise ValueError("Choisissez la colonne du rythme cardiaque fœtal (FHR).")
    fhr_brut = pd.to_numeric(df[col_fhr], errors="coerce").to_numpy(float)
    if np.isfinite(fhr_brut).sum() < 100:
        raise ValueError("La colonne FHR contient moins de 100 valeurs numériques : mauvaise colonne ?")
    fhr = fhr_brut.copy()
    fhr[~np.isfinite(fhr) | (fhr <= 0)] = np.nan
    mediane = np.nanmedian(fhr) if np.isfinite(fhr).any() else np.nan
    if not (60 <= mediane <= 220):
        av.append(f"Valeurs FHR inhabituelles (médiane = {mediane:.0f}) : un CTG est autour de 110–160 BPM. "
                  "Vérifiez la colonne choisie.")
    uc = None
    if col_uc and col_uc in df.columns:
        uc = pd.to_numeric(df[col_uc], errors="coerce").to_numpy(float)
        if not np.isfinite(uc).any():
            uc = None
            av.append("La colonne UC ne contient aucune valeur numérique : elle est ignorée.")

    if col_temps and col_temps in df.columns:
        t = pd.to_numeric(df[col_temps], errors="coerce").to_numpy(float)
        dt = np.diff(t)
        if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
            raise ValueError("La colonne temps est invalide (valeurs non numériques, répétées ou décroissantes).")
        dt_med = float(np.median(dt))
        unite = unite_temps
        if unite == "auto":                       # on devine d'après le pas : 4 Hz = 0,25 s = 250 ms = 0,0042 min
            unite = "ms" if dt_med >= 20 else ("min" if dt_med < 0.02 else "s")
            av.append(f"Unité du temps devinée : {unite} (modifiable dans la barre latérale).")
        t = t * {"s": 1.0, "min": 60.0, "ms": 1e-3}[unite]
        dt_med = float(np.median(np.diff(t)))
        fs = 1.0 / dt_med
        if np.mean(np.abs(np.diff(t) - dt_med) > 0.05 * dt_med) > 0.01:
            av.append("Échantillonnage irrégulier : FFT, STFT et ondelettes sont alors approximatives.")
        if not (0.5 <= fs <= 20):
            av.append(f"Fréquence d'échantillonnage inhabituelle ({fs:.3g} Hz) pour un CTG : vérifiez l'unité du temps.")
        reel = True
    else:
        t, fs, reel = np.arange(len(fhr)) / fs_defaut, float(fs_defaut), False
        av.append(f"Pas de colonne temps : fréquence d'échantillonnage supposée = {fs_defaut:g} Hz.")
    return Contexte(df, nom, t, fs, fhr_brut, fhr, uc, reel, av)


# ----------------------------------------------------------------------------
# Lecture de n'importe quel fichier : CSV, TXT, Excel, ou enregistrement PhysioNet (.hea + .dat)
# ----------------------------------------------------------------------------
def _octets(f):
    return f.getvalue() if hasattr(f, "getvalue") else f.read()


def _est_nombre(x):
    try:
        float(str(x).replace(",", "."))
        return True
    except ValueError:
        return False


def _lire_texte(octets):
    """Texte brut : détecte tout seul séparateur, en-tête, virgule décimale, lignes de commentaires."""
    texte = octets.decode("utf-8-sig", errors="replace")
    lignes = [l.strip() for l in texte.splitlines() if l.strip() and not l.lstrip().startswith(("#", "%", "//"))]
    if len(lignes) < 10:
        raise ValueError("Fichier vide ou trop court (minimum 10 lignes).")
    milieu = lignes[len(lignes) // 2: len(lignes) // 2 + 30]
    sep, trouve = None, False
    for s in (";", "\t", "|", ",", None):               # None = espaces
        nb = {len(l.split() if s is None else l.split(s)) for l in milieu}
        if len(nb) == 1 and nb != {1}:
            sep, trouve = s, True
            break
    if not trouve:
        raise ValueError("Séparateur de colonnes non reconnu (point-virgule, virgule, tabulation ou espace attendus).")
    rows = [l.split() if sep is None else [x.strip() for x in l.split(sep)] for l in lignes]
    n_col = len(rows[len(rows) // 2])
    corps = [r for r in rows if len(r) == n_col]
    entete = None
    if not all(_est_nombre(x) for x in corps[0]):
        entete, corps = corps[0], corps[1:]
    noms = []
    for i, n in enumerate(entete or [f"col_{i}" for i in range(n_col)]):
        n = n or f"col_{i}"
        noms.append(n if n not in noms else f"{n}_{i}")
    df = pd.DataFrame(corps, columns=noms)
    for c in df.columns:
        v = pd.to_numeric(df[c].str.replace(",", ".", regex=False) if sep != "," else df[c], errors="coerce")
        if v.notna().mean() > 0.5:
            df[c] = v
    return df


def _lire_wfdb(fichiers):
    try:
        import wfdb
    except ImportError:
        raise ValueError("Pour lire un enregistrement PhysioNet (.hea + .dat), installez wfdb : pip install wfdb")
    hea = [f for f in fichiers if f.name.lower().endswith(".hea")]
    dat = [f for f in fichiers if f.name.lower().endswith(".dat")]
    if not hea or not dat:
        raise ValueError("Un enregistrement PhysioNet a besoin des DEUX fichiers .hea et .dat : sélectionnez-les ensemble.")
    with tempfile.TemporaryDirectory() as d:
        for f in fichiers:
            with open(os.path.join(d, os.path.basename(f.name)), "wb") as out:
                out.write(_octets(f))
        base = os.path.join(d, os.path.splitext(os.path.basename(hea[0].name))[0])
        rec = wfdb.rdrecord(base)
    data = {"Time": np.arange(rec.sig_len) / rec.fs}
    for i, n in enumerate(rec.sig_name):
        data[n] = rec.p_signal[:, i]
    return pd.DataFrame(data), os.path.basename(hea[0].name)


def lire_fichiers(fichiers):
    """Accepte CSV, TXT, DAT texte, Excel ou enregistrement PhysioNet (.hea + .dat). Retourne (DataFrame, nom)."""
    if not isinstance(fichiers, (list, tuple)):
        fichiers = [fichiers]
    exts = {os.path.splitext(f.name)[1].lower() for f in fichiers}
    if ".hea" in exts or (".dat" in exts and len(fichiers) > 1):
        return _lire_wfdb(fichiers)
    f = fichiers[0]
    octets = _octets(f)
    try:
        if os.path.splitext(f.name)[1].lower() in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(BytesIO(octets))
            if all(_est_nombre(c) for c in df.columns):
                df = pd.read_excel(BytesIO(octets), header=None)
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
            df.columns = [str(c) for c in df.columns]
            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
        else:
            df = _lire_texte(octets)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Lecture impossible : {e}")
    if df.shape[0] < 10 or df.shape[1] < 1:
        raise ValueError("Le fichier contient trop peu de données.")
    return df, f.name


def lire_fichier(fichier):
    """Ancien nom conservé : un seul fichier -> DataFrame."""
    return lire_fichiers([fichier])[0]


def deviner_colonnes(df):
    """Devine (temps, FHR, UC) par les noms, puis par les valeurs (fonctionne même sans en-tête)."""
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in df.columns}
    num = {c: s for c, s in num.items() if s.notna().mean() > 0.5}
    motifs = {"t": r"time|temps|timestamp|\bsec|^t$|^t[ _(\[]", "f": r"fhr|fcf|bpm|heart|rythme|card|^fc\b",
              "u": r"\buc\b|^uc|toco|contr|ut[eé]r"}
    choix = {}
    for cle in ("f", "u", "t"):
        for c in num:
            if c not in choix.values() and re.search(motifs[cle], str(c).lower()):
                choix[cle] = c
                break
    libres = [c for c in num if c not in choix.values()]
    if "t" not in choix:                                   # colonne croissante à pas constant
        for c in libres:
            d = np.diff(num[c].dropna().to_numpy(float))
            if len(d) > 10 and np.all(d > 0) and np.std(d) < 0.05 * np.mean(d):
                choix["t"] = c
                libres.remove(c)
                break
    if "f" not in choix:                                   # valeurs plausibles de rythme cardiaque
        for c in libres:
            if 60 <= num[c].median() <= 220:
                choix["f"] = c
                libres.remove(c)
                break
    if "u" not in choix and libres:
        choix["u"] = libres[0]
    return choix.get("t"), choix.get("f"), choix.get("u")


# ----------------------------------------------------------------------------
# Petites fonctions utiles (utilisées par plusieurs parties)
# ----------------------------------------------------------------------------
def plus_long_segment(valide):
    """(début, fin) du plus long bloc de True dans un tableau booléen, ou None."""
    valide = np.asarray(valide, bool)
    if not valide.any():
        return None
    d = np.diff(np.r_[False, valide, False].astype(int))
    debuts, fins = np.where(d == 1)[0], np.where(d == -1)[0]
    i = int(np.argmax(fins - debuts))
    return int(debuts[i]), int(fins[i])


def interpoler(x):
    """Remplace les NaN par une interpolation linéaire (pour les détections qui exigent un signal continu)."""
    x = np.asarray(x, float).copy()
    mauvais = ~np.isfinite(x)
    if mauvais.any() and not mauvais.all():
        x[mauvais] = np.interp(np.flatnonzero(mauvais), np.flatnonzero(~mauvais), x[~mauvais])
    return x


def generer_signal_demo(duree_min=30, seed=0, retard_dec_s=20.0, fs=FS_DEFAUT):
    """Enregistrement SIMULÉ (pour tester sans fichier) : contractions, décélérations retardées, interruptions."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duree_min * 60, 1 / fs)
    tm = t / 60
    g = lambda c, s: np.exp(-0.5 * ((tm - c) / s) ** 2)
    contr = np.arange(4.0, duree_min - 2, 4.5)
    uc = 10 + sum(rng.uniform(60, 85) * g(c, 0.5) for c in contr) + rng.normal(0, 1.5, len(t))
    fhr = 140 + sum(20 * g(c + 2.25, 0.25) for c in contr[:-1]) \
        - sum(rng.uniform(22, 35) * g(c + retard_dec_s / 60, 0.38) for c in contr)
    bruit, e = np.zeros(len(t)), rng.normal(0, 0.9, len(t))
    for i in range(1, len(t)):
        bruit[i] = 0.9 * bruit[i - 1] + e[i]
    fhr = fhr + bruit
    for debut_min, long_s in ((6.3, 18), (14.1, 35), (22.7, 12)):      # pertes de signal (FHR = 0)
        if debut_min < duree_min - 1:
            i = int(debut_min * 60 * fs)
            fhr[i:i + int(long_s * fs)] = 0
    return pd.DataFrame({"Time": t.round(2), "FHR": fhr.round(1), "UC": uc.round(1)})


# ----------------------------------------------------------------------------
# Thème visuel « bébé » : pastel, bébé qui dort, cœur qui bat, tracé ECG
# ----------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp li, .stApp td, .stApp th,
.stApp button, .stApp input {font-family: 'Nunito', 'Segoe UI', sans-serif;}
.stApp {background: linear-gradient(180deg, #FFF7FA 0%, #F3F8FF 100%);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #FFEDF3 0%, #EAF3FF 100%);}
.block-container {padding-top: 1.2rem; max-width: 1250px;}

.hero {position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 20px;
       background: linear-gradient(120deg, #FFE4EC 0%, #E0F2FE 55%, #EDE9FE 100%);
       border: 3px solid #fff; border-radius: 28px; padding: 22px 34px 34px 34px; margin-bottom: 18px;
       box-shadow: 0 8px 28px rgba(244,63,94,.14);}
.hero-text {position: relative; z-index: 2;}
.hero-text h1 {color: #4A2C5A; margin: 0; padding: 0; font-size: 2.2rem; font-weight: 800;}
.hero-text p {color: #6B5B7B; margin: 6px 0 0 0; font-size: 1.04rem; font-weight: 600;}
.chips span {display: inline-block; background: #fff; color: #9D174D; border-radius: 999px; padding: 4px 13px;
             margin: 10px 8px 0 0; font-size: .82rem; font-weight: 800; box-shadow: 0 1px 5px rgba(0,0,0,.07);}
.bebe {width: 165px; flex-shrink: 0; position: relative; z-index: 2; animation: flotte 4.5s ease-in-out infinite;}
@keyframes flotte {0%,100% {transform: translateY(0);} 50% {transform: translateY(-9px);}}
.coeur {transform-box: fill-box; transform-origin: center; animation: bat 1.1s ease-in-out infinite;}
@keyframes bat {0%,100% {transform: scale(1);} 15% {transform: scale(1.28);} 30% {transform: scale(1);} 45% {transform: scale(1.18);}}
.etoile {animation: scintille 2.6s ease-in-out infinite;}
@keyframes scintille {0%,100% {opacity: .35;} 50% {opacity: 1;}}
.ecg {position: absolute; left: 0; bottom: 4px; width: 100%; height: 34px; z-index: 1; opacity: .75;}
.ecg-line {fill: none; stroke: #FB7185; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round;
           stroke-dasharray: 1800; stroke-dashoffset: 1800; animation: tracer 6s linear infinite;}
@keyframes tracer {to {stroke-dashoffset: 0;}}

.partie {background: #fff; border-left: 8px solid #F9A8D4; border-radius: 20px; padding: 14px 22px; margin: 4px 0 14px 0;
         box-shadow: 0 2px 12px rgba(74,44,90,.08);}
.partie h2 {margin: 6px 0 2px 0; padding: 0; font-size: 1.45rem; font-weight: 800; color: #4A2C5A;}
.partie p {margin: 0; color: #6B5B7B;}
.badge {background: #EDE9FE; color: #6D28D9; padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 800;}
div[data-testid="stMetric"] {background: #fff; border-radius: 18px; padding: 10px 15px; border-top: 4px solid #F9A8D4;
        box-shadow: 0 2px 10px rgba(74,44,90,.08);}
div[data-testid="stMetricLabel"] p {color: #7C6A8C; font-size: .82rem; font-weight: 700; white-space: normal; overflow: visible; text-overflow: clip;}
div[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 800; color: #4A2C5A;}
div[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
button[data-baseweb="tab"] {font-weight: 800;}
h3 {color: #4A2C5A;}
</style>
"""

_BEBE = """<svg class="bebe" viewBox="0 0 200 215" xmlns="http://www.w3.org/2000/svg">
<path class="etoile" d="M24 40 l3 8 8 3 -8 3 -3 8 -3 -8 -8 -3 8 -3z" fill="#FCD34D"/>
<path class="etoile" d="M172 98 l2.5 6 6 2.5 -6 2.5 -2.5 6 -2.5 -6 -6 -2.5 6 -2.5z" fill="#FCD34D" style="animation-delay:.9s"/>
<path class="etoile" d="M14 120 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z" fill="#C4B5FD" style="animation-delay:1.7s"/>
<path class="coeur" d="M164 34 c0 -9 -14 -11 -14 0 c0 8 14 16 14 16 s14 -8 14 -16 c0 -11 -14 -9 -14 0z" fill="#FB7185"/>
<path class="coeur" d="M30 82 c0 -6 -9 -7 -9 0 c0 5 9 10 9 10 s9 -5 9 -10 c0 -7 -9 -6 -9 0z" fill="#F9A8D4" style="animation-delay:.5s"/>
<ellipse cx="62" cy="14" rx="26" ry="9" fill="#fff" opacity=".85"/><ellipse cx="80" cy="9" rx="17" ry="9" fill="#fff" opacity=".85"/>
<path d="M30 215 Q100 148 170 215 Z" fill="#BFDBFE"/>
<circle cx="62" cy="196" r="4" fill="#fff" opacity=".8"/><circle cx="100" cy="186" r="4" fill="#fff" opacity=".8"/><circle cx="138" cy="198" r="4" fill="#fff" opacity=".8"/>
<circle cx="45" cy="112" r="10" fill="#FFD9C7"/><circle cx="155" cy="112" r="10" fill="#FFD9C7"/>
<circle cx="100" cy="108" r="58" fill="#FFE3D3"/>
<path d="M100 52 C88 33 114 30 109 49" stroke="#B9835F" stroke-width="4.5" fill="none" stroke-linecap="round"/>
<path d="M72 104 Q80 113 88 104" stroke="#5B3A4A" stroke-width="3.6" fill="none" stroke-linecap="round"/>
<path d="M112 104 Q120 113 128 104" stroke="#5B3A4A" stroke-width="3.6" fill="none" stroke-linecap="round"/>
<ellipse cx="65" cy="124" rx="11" ry="7" fill="#FDA4AF" opacity=".65"/><ellipse cx="135" cy="124" rx="11" ry="7" fill="#FDA4AF" opacity=".65"/>
<circle cx="100" cy="119" r="3.6" fill="#F2B3A0"/>
<path d="M92 134 Q100 143 108 134" stroke="#D9667A" stroke-width="3.6" fill="none" stroke-linecap="round"/>
</svg>"""

_ECG = ('<svg class="ecg" viewBox="0 0 1000 34" preserveAspectRatio="none"><polyline class="ecg-line" points="0,20 120,20 140,20 152,4 166,32 176,20 '
        '330,20 350,20 362,6 376,30 386,20 560,20 580,20 592,3 606,32 616,20 780,20 800,20 812,5 826,31 836,20 1000,20"/></svg>')


def appliquer_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def entete(titre, sous_titre):
    chips = '<div class="chips"><span>💗 110–160 battements/min</span><span>⏱ 4 mesures par seconde</span><span>📈 Signaux et Systèmes</span></div>'
    st.markdown(f'<div class="hero"><div class="hero-text"><h1>{titre}</h1><p>{sous_titre}</p>{chips}</div>{_BEBE}{_ECG}</div>',
                unsafe_allow_html=True)


def bandeau_partie(numero, titre, membre, resume):
    icone = {1: "🍼", 2: "🌊", 3: "🔗", 4: "🧸"}.get(numero, "🌟")
    badge = f"PARTIE {numero} · {membre}" if isinstance(numero, int) else membre
    st.markdown(f'<div class="partie"><span class="badge">{icone} {badge}</span>'
                f'<h2>{titre}</h2><p>{resume}</p></div>', unsafe_allow_html=True)


def notions(titre, texte):
    with st.expander(f"📘 Notions de Signaux et Systèmes — {titre}"):
        st.markdown(texte)


def layout(fig, titre="", hauteur=420, hover="x unified"):
    fig.update_layout(template="plotly_white", title=dict(text=titre, x=0.01, font=dict(size=16, color="#4A2C5A")),
                      height=hauteur, margin=dict(l=10, r=10, t=55 if titre else 30, b=10),
                      legend=dict(orientation="h", y=1.1, x=0), hovermode=hover,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.75)",
                      font=dict(family="Nunito, Segoe UI, sans-serif", color="#4A2C5A"))
    fig.update_xaxes(gridcolor="#F1E4EC")
    fig.update_yaxes(gridcolor="#F1E4EC")
    return fig
