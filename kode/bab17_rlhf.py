"""Bab 17: model imbalan Bradley-Terry, RLHF dengan penalti KL, dan DPO
pada 'model bahasa' mini: semua kalimat tiga token dari {a, b, c}.

Kebijakan = sebaran softmax atas 27 kalimat (tanpa konteks). Kebijakan
acuan pi_ref menyukai b. Imbalan sebenarnya: banyaknya a, kecuali
kalimat berulang "aaa" yang bernilai 3 - 4 = -1. Model imbalan hanya
melihat banyaknya a, b, c, sehingga tidak dapat menangkap hukuman itu.

Dipakai bab-bab berikutnya:
  KALIMAT, fitur(y), r_benar(y), PI_REF
  bt_peluang(r1, r2), optimum_kl(pi_ref, r, beta)
"""
import itertools

import numpy as np

BENIH = 20261009
KALIMAT = ["".join(k) for k in itertools.product("abc", repeat=3)]
FITUR = np.array([[y.count(c) for c in "abc"] for y in KALIMAT], float)


def r_benar(y):
    return y.count("a") - 4.0 * (y == "aaa")


R_BENAR = np.array([r_benar(y) for y in KALIMAT])
_p = np.array([0.25, 0.5, 0.25])                 # peluang token acuan
PI_REF = np.array([np.prod([_p["abc".index(c)] for c in y])
                   for y in KALIMAT])


def bt_peluang(r1, r2):
    return 1.0 / (1.0 + np.exp(-(r1 - r2)))


def optimum_kl(pi_ref, r, beta):
    """argmax E_pi[r] - beta KL(pi || pi_ref) = pi_ref exp(r/beta) / Z."""
    w = np.log(pi_ref) + r / beta
    w -= w.max()
    p = np.exp(w)
    return p / p.sum()


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def data_preferensi(n, rng):
    i = rng.choice(27, size=n, p=PI_REF)
    j = rng.choice(27, size=n, p=PI_REF)
    pilih_i = rng.random(n) < bt_peluang(R_BENAR[i], R_BENAR[j])
    menang = np.where(pilih_i, i, j)
    kalah = np.where(pilih_i, j, i)
    return menang, kalah


def latih_model_imbalan(menang, kalah, iterasi=3000, lr=0.5):
    """Regresi logistik Bradley-Terry: r_hat(y) = w . fitur(y)."""
    w = np.zeros(3)
    D = FITUR[menang] - FITUR[kalah]
    for _ in range(iterasi):
        p = 1 / (1 + np.exp(-D @ w))
        w += lr * D.T @ (1 - p) / len(D)
    return w


def latih_dpo(menang, kalah, beta, iterasi=4000, lr=2.0):
    """DPO: logit kebijakan theta (27), pi = softmax(theta)."""
    theta = np.log(PI_REF).copy()
    for _ in range(iterasi):
        logpi = theta - np.log(np.exp(theta - theta.max()).sum()) - theta.max()
        h = beta * ((logpi[menang] - np.log(PI_REF[menang]))
                    - (logpi[kalah] - np.log(PI_REF[kalah])))
        g_h = -(1 - 1 / (1 + np.exp(-h)))            # turunan -log sigma(h)
        pi = np.exp(logpi)
        grad = np.zeros(27)
        np.add.at(grad, menang, beta * g_h)
        np.add.at(grad, kalah, -beta * g_h)
        grad -= pi * grad.sum()                       # rantai lewat softmax
        theta -= lr * grad / len(menang)
    p = np.exp(theta - theta.max())
    return p / p.sum()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(BENIH)
    menang, kalah = data_preferensi(1000, rng)
    w = latih_model_imbalan(menang, kalah)
    r_hat = FITUR @ w
    print("(1) Model imbalan Bradley-Terry dari 1000 pasangan")
    print("    bobot (a, b, c) =", w)
    print(f"    E_ref[r benar] = {PI_REF @ R_BENAR:.4f}")
    print("(2) RLHF (optimum KL dengan r_hat) lawan DPO, data yang sama")
    print("    beta   metode   KL     r_hat   r benar")
    for beta in (2.0, 1.0, 0.5, 0.25, 0.1, 0.05):
        pr = optimum_kl(PI_REF, r_hat, beta)
        pd = latih_dpo(menang, kalah, beta)
        for nama, p in (("RLHF", pr), ("DPO", pd)):
            print(f"    {beta:4.2f}   {nama:6s} {kl(p, PI_REF):5.2f}"
                  f" {p @ r_hat:7.3f} {p @ R_BENAR:8.3f}")
    for beta in (0.25, 0.05):
        pb = optimum_kl(PI_REF, R_BENAR, beta)
        print(f"    {beta:4.2f}   r*     {kl(pb, PI_REF):5.2f}"
              f"       - {pb @ R_BENAR:8.3f}")
    pr = optimum_kl(PI_REF, r_hat, 0.05)
    print(f"    RLHF beta 0.05: P(aaa) = {pr[0]:.3f}")
