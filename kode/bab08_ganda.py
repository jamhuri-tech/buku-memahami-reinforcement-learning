"""Bab 8: bias maksimisasi dan double Q-learning.

MDP dua keadaan: dari A, aksi kanan selesai dengan imbalan 0, aksi kiri
pindah ke B dengan imbalan 0. Dari B, setiap aksi dari M aksi selesai
dengan imbalan N(-0.1, 1). Optimal: selalu kanan di A (nilai 0 > -0.1).
Semua ulangan dijalankan serentak (vektor).
"""
import numpy as np

from bab01_lorong import BENIH


def jalankan(ganda, n_ep=300, R=2000, M=10, alpha=0.1, eps=0.1,
             benih=BENIH):
    rng = np.random.default_rng(benih)
    rng_l = np.random.default_rng(benih + 1)
    QA = np.zeros((2, R, 2))           # dua tabel untuk A (ganda)
    QB = np.zeros((2, R, M))
    baris = np.arange(R)
    kiri = np.zeros(n_ep)

    def pilih(q):
        acak = rng.random(R) < eps
        seri = rng.random(q.shape) * (q == q.max(1, keepdims=True))
        a = seri.argmax(1)
        a[acak] = rng.integers(q.shape[1], size=acak.sum())
        return a

    for e in range(n_ep):
        qA = QA.sum(0) if ganda else QA[0]
        a = pilih(qA)                   # 0 = kiri (ke B), 1 = kanan
        kiri[e] = (a == 0).mean()
        u = (rng.random(R) < 0.5).astype(int) if ganda else np.zeros(R, int)
        v = 1 - u
        # langkah dari A
        keB = a == 0
        if ganda:
            a_b = QB[u, baris].argmax(1)                  # pilih dengan u
            nilai_B = QB[v, baris, a_b]                   # nilai dari v
        else:
            nilai_B = QB[0].max(1)
        target = np.where(keB, nilai_B, 0.0)
        QA[u, baris, a] += alpha * (target - QA[u, baris, a])
        # langkah dari B (hanya ulangan yang ke B)
        qB = QB.sum(0) if ganda else QB[0]
        b = pilih(qB)
        r = rng_l.normal(-0.1, 1.0, R)
        u2 = (rng.random(R) < 0.5).astype(int) if ganda else np.zeros(R, int)
        i = np.where(keB)[0]
        QB[u2[i], i, b[i]] += alpha * (r[i] - QB[u2[i], i, b[i]])
    return kiri


if __name__ == "__main__":
    print("(3) Bias maksimisasi: bagian ulangan yang memilih kiri di A")
    print("    optimal (eps-greedy, eps = 0.1) = 0.05")
    print("    episode   Q-learning   double Q")
    q = jalankan(False)
    d = jalankan(True)
    for e in (1, 5, 10, 20, 50, 100, 300):
        print(f"    {e:7d} {q[e - 1]:12.3f} {d[e - 1]:10.3f}")
