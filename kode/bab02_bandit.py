"""Bab 2: bandit multi-lengan Bernoulli.

Dipakai bab-bab berikutnya:
  MU_MINI                peluang sukses tiga lengan data mini
  RIWAYAT                delapan tarikan (lengan, imbalan) Contoh Soal
  rata_inkremental(...)  Q_{n+1} = Q_n + alpha (R_n - Q_n)
  argmax_acak(Q, rng)    argmax dengan seri dipecah secara acak
  simulasi(...)          banyak ulangan sekaligus, satu strategi
"""
import numpy as np

BENIH = 20261009

MU_MINI = np.array([0.25, 0.50, 0.75])
# (lengan, imbalan); lengan bernomor 1, 2, 3 seperti di naskah
RIWAYAT = [(1, 0), (2, 1), (3, 1), (1, 1), (2, 0), (3, 0), (3, 1), (3, 1)]


def rata_inkremental(imbalan, alpha=None, q0=0.0):
    """Barisan Q_1, Q_2, ...; alpha None berarti 1/n."""
    q, hasil = q0, []
    for n, r in enumerate(imbalan, 1):
        a = 1.0 / n if alpha is None else alpha
        q = q + a * (r - q)
        hasil.append(q)
    return hasil


def argmax_acak(Q, rng):
    """argmax per baris; nilai yang seri dipilih secara acak."""
    Q = np.atleast_2d(Q)
    maks = Q.max(axis=1, keepdims=True)
    acak = rng.random(Q.shape) * (Q == maks)
    return acak.argmax(axis=1)


def simulasi(mu, strategi, T=1000, R=2000, benih=BENIH, tarik=None,
             **kw):
    """R ulangan bandit sekaligus; imbalan Bernoulli(mu) kecuali
    tarik(a, rng) diberikan (imbalan lain di [0, 1], rata-rata mu).

    strategi: "greedy", "greedy-argmax", "eps", "optimis", "ucb",
    "thompson". Mengembalikan regret kumulatif rata-rata (T,) dan
    bagian ulangan yang memilih lengan terbaik pada setiap t.
    """
    rng = np.random.default_rng(benih)          # pilihan agen
    rng_l = np.random.default_rng(benih + 1)    # imbalan lingkungan
    mu = np.asarray(mu, float)
    k = len(mu)
    N = np.zeros((R, k))
    Q = np.full((R, k), kw.get("q0", 0.0))
    S = np.zeros((R, k))                         # banyaknya sukses
    baris = np.arange(R)
    terbaik = mu.argmax()
    regret = np.zeros(T)
    tepat = np.zeros(T)
    for t in range(T):
        if strategi == "greedy-argmax":
            a = Q.argmax(axis=1)
        elif strategi in ("greedy", "optimis"):
            a = argmax_acak(Q, rng)
        elif strategi == "eps":
            a = argmax_acak(Q, rng)
            jelajah = rng.random(R) < kw["eps"]
            a[jelajah] = rng.integers(k, size=jelajah.sum())
        elif strategi == "ucb":
            if t < k:
                a = np.full(R, t)
            else:
                bonus = np.sqrt(kw.get("c", 2.0) * np.log(t) / N)
                a = argmax_acak(Q + bonus, rng)
        elif strategi == "thompson":
            a = rng.beta(1 + S, 1 + N - S).argmax(axis=1)
        if tarik is None:
            r = (rng_l.random(R) < mu[a]).astype(float)
        else:
            r = tarik(a, rng_l)
        N[baris, a] += 1
        S[baris, a] += r
        alpha = kw.get("alpha")
        langkah = 1.0 / N[baris, a] if alpha is None else alpha
        Q[baris, a] += langkah * (r - Q[baris, a])
        regret[t] = (mu[terbaik] - mu[a]).mean()
        tepat[t] = (a == terbaik).mean()
    return np.cumsum(regret), tepat


STRATEGI = [
    ("greedy", "greedy", {}),
    ("eps = 0,1", "eps", {"eps": 0.1}),
    ("eps = 0,01", "eps", {"eps": 0.01}),
    ("optimis Q0=1", "optimis", {"q0": 1.0, "alpha": 0.1}),
    ("UCB1", "ucb", {"c": 2.0}),
    ("Thompson", "thompson", {}),
]

if __name__ == "__main__":
    print("(1) Bandit Bernoulli, T = 1000, rata-rata 2000 ulangan:")
    for mu in (MU_MINI, np.array([0.45, 0.50, 0.55])):
        print(f"    mu = {tuple(float(m) for m in mu)}")
        print("      strategi        regret T=1000   lengan terbaik"
              " (t>900)")
        for nama, s, kw in STRATEGI:
            reg, tepat = simulasi(mu, s, **kw)
            print(f"      {nama:15s} {reg[-1]:12.2f}"
                  f" {100 * tepat[900:].mean():14.1f} %")

    print("(2) Greedy, seri dipecah np.argmax lawan dipecah acak:")
    for nama, s in (("np.argmax", "greedy-argmax"),
                    ("acak", "greedy")):
        reg, tepat = simulasi(MU_MINI, s)
        print(f"    {nama:10s} lengan terbaik {100 * tepat[-1]:5.1f} %"
              f"   regret {reg[-1]:7.2f}")

    print("(3) Thompson, posterior Beta(2,2), Beta(2,2), Beta(4,2):")
    rng = np.random.default_rng(BENIH)
    th = rng.beta([2, 2, 4], [2, 2, 2], size=(1_000_000, 3))
    p = np.bincount(th.argmax(1), minlength=3) / len(th)
    print("    peluang setiap lengan dipilih =", np.round(p, 4))
    print(f"    P(theta3 > theta1) = {(th[:, 2] > th[:, 0]).mean():.4f}"
          f" (eksak 5/7 = {5 / 7:.4f})")
