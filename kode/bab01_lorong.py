"""Bab 1: data mini lorong licin, episode, return, dan nilai kebijakan.

Dipakai bab-bab berikutnya:
  KIRI, KANAN, AKSI       aksi 0 dan 1
  KEADAAN, TERMINAL       keadaan tak-terminal 1, 2, 3 dan terminal 0, 4
  GAMMA                   faktor diskon data mini (1/2)
  Lorong(licin, benih)    lingkungan gaya Gymnasium: reset(), step(a)
  dinamika(s, a)          daftar (peluang, s_baru, imbalan) yang eksak
  kebijakan_det(aksi)     kebijakan deterministik dari tiga aksi
  ACAK                    kebijakan acak seragam
  jalankan_episode(...)   satu episode: daftar (s, a, r)
  return_(imbalan, g)     G_0, G_1, ... dari daftar imbalan
  nilai_eksak(pi, g)      V^pi untuk keadaan 1, 2, 3 (Bab 4)
"""
import numpy as np

BENIH = 20261009

KIRI, KANAN = 0, 1
AKSI = (KIRI, KANAN)
NAMA_AKSI = ("kiri", "kanan")
KEADAAN = (1, 2, 3)
TERMINAL = (0, 4)
AWAL = 2
GAMMA = 0.5
LICIN = 0.25              # peluang tergelincir di keadaan 2
IMBALAN = {0: 2.0, 4: 8.0}  # imbalan saat masuk ke keadaan terminal


def dinamika(s, a, licin=LICIN):
    """p(s', r | s, a) untuk s = 1, 2, 3, sebagai daftar tiga-an."""
    arah = -1 if a == KIRI else 1
    if s == 2 and licin > 0:
        keluar = [(1 - licin, s + arah), (licin, s - arah)]
    else:
        keluar = [(1.0, s + arah)]
    return [(p, s2, IMBALAN.get(s2, 0.0)) for p, s2 in keluar]


class Lorong:
    """Lorong licin lima petak. step(a) -> (s_baru, r, selesai)."""

    def __init__(self, licin=LICIN, benih=BENIH):
        self.licin = licin
        self.rng = np.random.default_rng(benih)
        self.s = AWAL

    def reset(self, s=AWAL):
        self.s = s
        return self.s

    def step(self, a):
        keluar = dinamika(self.s, a, self.licin)
        u = self.rng.random()
        for p, s2, r in keluar:
            if u < p:
                break
            u -= p
        self.s = s2
        return s2, r, s2 in TERMINAL


def kebijakan_det(aksi):
    """aksi = (a1, a2, a3): aksi di keadaan 1, 2, 3."""
    pi = np.zeros((5, 2))
    for s, a in zip(KEADAAN, aksi):
        pi[s, a] = 1.0
    return pi


ACAK = np.full((5, 2), 0.5)
KANAN_SEMUA = kebijakan_det((KANAN, KANAN, KANAN))
OPTIMUM = kebijakan_det((KIRI, KANAN, KANAN))


def jalankan_episode(env, pi, rng, s0=AWAL, maks=10_000):
    """Mengembalikan daftar (S_t, A_t, R_{t+1})."""
    s, jejak = env.reset(s0), []
    for _ in range(maks):
        a = rng.choice(2, p=pi[s])
        s2, r, selesai = env.step(a)
        jejak.append((s, a, r))
        s = s2
        if selesai:
            break
    return jejak


def return_(imbalan, g=GAMMA):
    """G_t = R_{t+1} + g G_{t+1}, dihitung mundur dari akhir."""
    G, hasil = 0.0, []
    for r in reversed(imbalan):
        G = r + g * G
        hasil.append(G)
    return hasil[::-1]


def nilai_eksak(pi, g=GAMMA, licin=LICIN):
    """V^pi(1..3) dari sistem linear (I - g P) v = r (Bab 4)."""
    P = np.zeros((3, 3))
    rv = np.zeros(3)
    for i, s in enumerate(KEADAAN):
        for a in AKSI:
            for p, s2, r in dinamika(s, a, licin):
                w = pi[s, a] * p
                rv[i] += w * r
                if s2 not in TERMINAL:
                    P[i, s2 - 1] += w * g
    return np.linalg.solve(np.eye(3) - P, rv)


if __name__ == "__main__":
    # Agen dan lingkungan memakai aliran acak yang terpisah.
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)

    print("(1) Satu episode kebijakan acak seragam, gamma = 1/2:")
    jejak = jalankan_episode(env, ACAK, rng)
    G = return_([r for _, _, r in jejak])
    print("     t  S_t  A_t    R_t+1    G_t")
    for t, ((s, a, r), g) in enumerate(zip(jejak, G)):
        print(f"    {t:2d}  {s:3d}  {NAMA_AKSI[a]:5s}  {r:5.0f}  {g:7.4f}")

    print("(2) V(2): rata-rata 100 000 episode lawan nilai eksak:")
    print("    kebijakan         rata-rata  galat baku   eksak"
          "  langkah")
    for nama, pi in (("acak seragam", ACAK),
                     ("selalu kanan", KANAN_SEMUA),
                     ("kiri di 1", OPTIMUM)):
        g0, panjang = [], []
        for _ in range(100_000):
            j = jalankan_episode(env, pi, rng)
            g0.append(return_([r for _, _, r in j])[0])
            panjang.append(len(j))
        g0 = np.array(g0)
        gb = g0.std(ddof=1) / np.sqrt(len(g0))
        print(f"    {nama:16s} {g0.mean():9.4f} {gb:10.4f}"
              f" {nilai_eksak(pi)[1]:8.4f} {np.mean(panjang):7.2f}")

    print("(3) Kebijakan acak seragam, benih agen = benih lingkungan:")
    rng_a = np.random.default_rng(BENIH)
    env_a = Lorong(benih=BENIH)
    g0 = [return_([r for _, _, r in jalankan_episode(env_a, ACAK,
                                                      rng_a)])[0]
          for _ in range(100_000)]
    print(f"    rata-rata G_0 = {np.mean(g0):.4f}"
          f" (eksak {nilai_eksak(ACAK)[1]:.4f})")

    print("(4) Delapan kebijakan (- kiri, + kanan) di keadaan 1, 2, 3:")
    for g in (0.5, 0.9):
        hasil = []
        for a1 in AKSI:
            for a2 in AKSI:
                for a3 in AKSI:
                    v = nilai_eksak(kebijakan_det((a1, a2, a3)), g) + 0.0
                    hasil.append(("".join("-+"[a] for a in (a1, a2, a3)),
                                  v))
        terbaik = max(hasil, key=lambda h: h[1].sum())[0]
        print(f"    gamma = {g}:")
        for kode, v in hasil:
            tanda = "  <- terbaik" if kode == terbaik else ""
            print(f"      {kode}  {v[0]:7.4f} {v[1]:7.4f}"
                  f" {v[2]:7.4f}{tanda}")
