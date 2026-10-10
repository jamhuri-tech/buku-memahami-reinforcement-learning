"""Bab 3: pengamatan tak lengkap dan pembentukan imbalan.

Dipakai bab-bab berikutnya:
  bentuk_potensial(mdp, Phi)  MDP dengan F = gamma Phi(s') - Phi(s)
  bentuk_bonus(mdp, s, b)     MDP dengan bonus b setiap masuk ke s
"""
import numpy as np

from bab03_mdp import MDP, mdp_lorong, semua_det


def bentuk_potensial(mdp, Phi):
    Phi = np.asarray(Phi, float)
    F = mdp.gamma * Phi[None, None, :] - Phi[:, None, None]
    F[mdp.terminal] = 0.0
    return MDP(mdp.P, mdp.R + F, mdp.gamma, terminal=np.where(
        mdp.terminal)[0], awal=mdp.awal)


def bentuk_bonus(mdp, s_bonus, b):
    B = np.zeros_like(mdp.R)
    B[:, :, s_bonus] = b
    B[s_bonus, :, s_bonus] = 0.0
    B[mdp.terminal] = 0.0
    return MDP(mdp.P, mdp.R + B, mdp.gamma, terminal=np.where(
        mdp.terminal)[0], awal=mdp.awal)


def terbaik(mdp):
    hasil = [(k, mdp.nilai(pi)) for k, pi in semua_det(mdp)]
    return max(hasil, key=lambda h: h[1].sum())


if __name__ == "__main__":
    m = mdp_lorong()
    print("(3) Agen hanya melihat tengah (petak 2) atau pinggir"
          " (1, 3):")
    print("    tengah pinggir  setara   V(2)")
    for at in (0, 1):
        for ap in (0, 1):
            kode = "-+"[ap] + "-+"[at] + "-+"[ap]
            pi = dict(semua_det(m))[kode]
            print(f"    {('kiri', 'kanan')[at]:6s} {('kiri', 'kanan')[ap]:7s}"
                  f"  {kode:6s} {m.nilai(pi)[2]:7.4f}")
    qs = np.linspace(0, 1, 101)
    v = [m.nilai(np.array([[1, 0], [1 - q, q], [0, 1], [1 - q, q],
                           [1, 0]]))[2] for q in qs]
    i = int(np.argmax(v))
    print(f"    acak di pinggir: maks V(2) = {v[i]:.4f},"
          f" P(kanan) = {qs[i]:.2f}")
    print(f"    keadaan lengkap (-++): V(2) = "
          f"{m.nilai(dict(semua_det(m))['-++'])[2]:.4f}")

    print("(4) Pembentukan imbalan, gamma = 1/2: kebijakan terbaik")
    print("    imbalan         pi    V dibentuk (1,2,3)"
          "     V asli")
    skema = [("asli", m),
             ("potensial Phi=s", bentuk_potensial(m, [0, 1, 2, 3, 0])),
             ("bonus masuk 2", bentuk_bonus(m, 2, 1.0))]
    for nama, mb in skema:
        kode, v = terbaik(mb)
        v_asli = m.nilai(dict(semua_det(m))[kode])
        print(f"    {nama:15s} {kode:4s}"
              + "".join(f" {x:5.2f}" for x in v[1:4])
              + "  " + "".join(f" {x:5.2f}" for x in v_asli[1:4]))
