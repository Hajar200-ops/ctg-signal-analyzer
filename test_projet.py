"""Tests du projet — un bloc par membre. Lancer :  pip install pytest   puis   pytest -v

Chaque test utilise un signal dont on connaît la réponse à l'avance : si le test passe,
la partie correspondante calcule bien ce qu'elle prétend calculer.
"""
import io

import numpy as np
import pandas as pd
import pytest

import commun as C
import partie1_acquisition as P1
import partie2_frequentiel as P2
import partie3_correlations as P3
import partie4_tempsfreq_ml as P4

FS = 4.0


def _ctx(duree_min=40, **kw):
    df = C.generer_signal_demo(duree_min=duree_min, **kw)
    return C.construire_contexte(df, "Time", "FHR", "UC", "test")


class Fichier(io.BytesIO):                      # imite un fichier déposé dans l'interface
    def __init__(self, contenu, nom):
        super().__init__(contenu)
        self.name = nom


# =========================================================== MEMBRE 1 : import, détection, indicateurs
def test_m1_csv_standard():
    df = C.generer_signal_demo(duree_min=10)
    d, nom = C.lire_fichiers(Fichier(df.to_csv(index=False).encode(), "a.csv"))
    assert C.deviner_colonnes(d) == ("Time", "FHR", "UC")


def test_m1_point_virgule_et_virgule_decimale():
    df = C.generer_signal_demo(duree_min=10)
    d, _ = C.lire_fichiers(Fichier(df.to_csv(index=False, sep=";", decimal=",").encode(), "b.csv"))
    ctx = C.construire_contexte(d, *C.deviner_colonnes(d), "b")
    assert abs(ctx.fs - FS) < 0.1 and np.nanmedian(ctx.fhr) > 100


def test_m1_colonnes_sans_en_tete_ni_noms():
    df = C.generer_signal_demo(duree_min=10)
    contenu = df[["UC", "Time", "FHR"]].to_csv(index=False, header=False).encode()   # ordre mélangé
    d, _ = C.lire_fichiers(Fichier(contenu, "c.csv"))
    t, f, u = C.deviner_colonnes(d)
    assert (t, f, u) == ("col_1", "col_2", "col_0")


def test_m1_temps_en_minutes_et_en_ms():
    df = C.generer_signal_demo(duree_min=10)
    for facteur, nom in ((1 / 60, "min"), (1000, "ms")):
        d = pd.DataFrame({"t": df["Time"] * facteur, "fhr": df["FHR"], "uc": df["UC"]})
        ctx = C.construire_contexte(d, "t", "fhr", "uc", nom)
        assert abs(ctx.fs - FS) < 0.2, nom


def test_m1_fichier_invalide_message_clair():
    with pytest.raises(ValueError):
        C.lire_fichiers(Fichier(b"bonjour\n" * 30, "x.csv"))


def test_m1_aucun_filtrage():
    """Le FHR analysé est exactement celui du fichier (seuls les 0 / manquants deviennent NaN)."""
    ctx = _ctx(10)
    valide = np.isfinite(ctx.fhr)
    assert np.array_equal(ctx.fhr[valide], ctx.fhr_brut[valide])


def test_m1_detection_evenements():
    ctx = _ctx(40)
    prm = {"seuil": 15.0, "duree_min": 10.0, "dist_fhr_s": 20.0, "seuil_uc": 0.3, "dist_uc_s": 90.0,
           "duree_min_uc": 30.0, "gap_max_s": 15.0, "fenetre_assoc_s": 120.0}
    base, _, acc, dec, contr = P1.detecter(ctx, prm)
    assert 135 <= base <= 145
    assert len(contr) == 8 and len(dec) >= 6
    assert all(10 <= e["decalage"] <= 30 for e in dec if e["decalage"] is not None)   # simulé avec 20 s de retard


def test_m1_indicateurs_signalent_decelerations_retardees():
    res = P1.resultats(_ctx(40))
    ligne = next(l for l in res["indicateurs"] if l[0] == "Décélérations")
    assert "retardées après" in ligne[3]


def test_m1_repliement_spectral():
    """Aliasing : un sinus de 0,8 Hz sous-échantillonné à 0,25 Hz (M = 16, SANS filtre) apparaît à 0,05 Hz."""
    t = np.arange(0, 4000, 1 / FS)
    x = np.sin(2 * np.pi * 0.8 * t)
    f, p = P1._psd(x[::16], FS / 16)
    assert f[np.argmax(p)] == pytest.approx(0.05, abs=0.005)           # 0,8 − 3×0,25 = 0,05 Hz


# =========================================================== MEMBRE 2 : FFT, Parseval, Welch, STFT
def test_m2_fft_frequence_sinus():
    t = np.arange(0, 1200, 1 / FS)
    f, a = P2.spectre(140 + 10 * np.sin(2 * np.pi * 0.01 * t), FS)
    assert P2.frequence_dominante(f, a, len(t)) == pytest.approx(0.01, rel=0.05)


def test_m2_amplitude_du_spectre():
    """Un sinus d'amplitude 10 BPM doit donner un pic d'amplitude ≈ 10."""
    t = np.arange(0, 1200, 1 / FS)
    f, a = P2.spectre(140 + 10 * np.sin(2 * np.pi * 0.0125 * t), FS)
    assert a.max() == pytest.approx(10, rel=0.1)


def test_m2_parseval():
    x = np.random.default_rng(0).normal(size=2048)
    assert np.sum(x ** 2) == pytest.approx(np.sum(np.abs(np.fft.fft(x)) ** 2) / len(x), rel=1e-10)


def test_m2_periode_des_contractions():
    res = P2.resultats(_ctx(40))
    assert 1 / res["f_dom_uc"] / 60 == pytest.approx(4.5, abs=0.5)      # une contraction toutes les 4,5 min


# =========================================================== MEMBRE 3 : autocorrélation, corrélation croisée
def test_m3_autocorrelation_periode():
    t = np.arange(0, 3000, 1 / FS)
    _, _, per = P3.autocorrelation(np.sin(2 * np.pi * t / 300), FS)
    assert per == pytest.approx(300, rel=0.03)


def test_m3_correlation_croisee_decalage():
    ctx = _ctx(40)
    u = np.nan_to_num(ctx.uc)
    retard = int(30 * FS)
    fhr = 140 - 0.5 * np.r_[np.full(retard, u[0]), u[:-retard]]          # FHR = −UC retardé de 30 s
    _, _, tau, coef = P3.correlation_croisee(fhr, u, FS, 120)
    assert tau == pytest.approx(30, abs=1.5) and coef < -0.9


def test_m3_correlation_signal_constant():
    assert P3.correlation_croisee(np.ones(500), np.arange(500.0), FS, 60) is None


def test_m3_resultats_coherents():
    res = P3.resultats(_ctx(40))
    assert res["periode_auto_uc_s"] / 60 == pytest.approx(4.5, abs=0.6)
    assert res["coef"] < -0.5 and 5 < res["tau_s"] < 40


# =========================================================== MEMBRE 4 : fenêtres, ondelettes, anomalies
def test_m4_fenetres_nombre_et_disponibilite():
    ctx = _ctx(10)
    df = P4.fenetres(ctx.t, ctx.uc, FS, 30, lambda w: np.isfinite(w) & (w >= 0), "UC")
    assert len(df) == 20                                                    # 10 min / 30 s
    assert (df["Disponibilité UC (%)"] >= 80).all()


def test_m4_fenetre_rejetee_si_trop_de_trous():
    x = np.full(1200, 140.0)
    x[:400] = np.nan                                                         # 1re fenêtre de 30 s : ≥ 80 % de trous
    df = P4.fenetres(np.arange(1200) / FS, x, FS, 30, lambda w: np.isfinite(w), "FHR")
    assert len(df) < 10


def test_m4_ondelettes_frequence_sinus():
    t = np.arange(0, 600, 1 / FS)
    mag, f = P4._cwt(np.sin(2 * np.pi * 0.05 * t), FS, 0.01, 1.0, 60)
    assert f[np.argmax(mag.mean(axis=1))] == pytest.approx(0.05, rel=0.2)


def test_m4_energie_centree():
    x = np.array([1.0, 2.0, 3.0, 4.0] * 30)
    df = P4.fenetres(np.arange(120) / FS, x, FS, 30, lambda w: np.isfinite(w), "FHR")
    assert df["Énergie centrée FHR"].iloc[0] == pytest.approx(np.sum((x - x.mean()) ** 2))
