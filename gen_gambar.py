# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261009  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().


# ============================ Bab 1 ==================================

def _kotak(ax, x, y, teks, w=2.6, h=0.62, wr=BIRU, isi=BIRU_MUDA, fs=6.3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=isi, edgecolor=wr, lw=0.7))
    ax.text(x, y, teks, ha="center", va="center", fontsize=fs)


def _panah(ax, a, b, wr=ABU, ls="-", rad=0.0, lw=0.7):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                 color=wr, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def bab01_siklus():
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    _kotak(ax, 5, 3.6, "agen\n" r"memilih $A_t \sim \pi(\cdot \mid S_t)$",
           w=3.6, h=0.95)
    _kotak(ax, 5, 0.8, "lingkungan\n"
           r"$S_{t+1}, R_{t+1} \sim p(\cdot, \cdot \mid S_t, A_t)$",
           w=4.4, h=0.95, wr=HIJAU, isi=HIJAU_MUDA)
    _panah(ax, (6.8, 3.6), (7.4, 0.8), wr=BIRU, rad=-0.5)
    ax.text(8.55, 2.2, r"aksi $A_t$", fontsize=6.5, color=BIRU)
    _panah(ax, (2.8, 1.0), (3.2, 3.4), wr=HIJAU, rad=-0.45)
    _panah(ax, (2.8, 0.6), (3.2, 3.8), wr=JINGGA, rad=-0.6)
    ax.text(0.05, 2.45, r"keadaan $S_{t+1}$", fontsize=6.5, color=HIJAU)
    ax.text(0.05, 1.85, r"imbalan $R_{t+1}$", fontsize=6.5, color=JINGGA)
    simpan(fig, "bab01-siklus")


def bab01_lorong():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.4, 1.55))
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.95, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    for s in range(5):
        terminal = s in (0, 4)
        isi = JINGGA_MUDA if terminal else (MERAH_MUDA if s == 2
                                             else BIRU_MUDA)
        ax.add_patch(Rectangle((s - 0.42, -0.42), 0.84, 0.84,
                               facecolor=isi, edgecolor=ABU, lw=0.6))
        ax.text(s, 0.08, str(s), ha="center", va="center", fontsize=9,
                color=BIRU)
    ax.text(0, -0.24, "$+2$", ha="center", fontsize=6.5, color=JINGGA)
    ax.text(4, -0.24, "$+8$", ha="center", fontsize=6.5, color=JINGGA)
    ax.text(2, -0.26, "licin", ha="center", fontsize=6, color=MERAH)
    ax.text(2, 0.29, "awal", ha="center", va="center", fontsize=5.5,
            color=ABU)
    _panah(ax, (2.15, 0.48), (2.85, 0.48), wr=BIRU, rad=-0.35)
    ax.text(2.5, 0.86, r"$\frac{3}{4}$", ha="center", fontsize=6.5,
            color=BIRU)
    _panah(ax, (2.15, 0.48), (1.15, 0.48), wr=MERAH, rad=0.45, ls="--")
    ax.text(1.5, 0.92, r"$\frac{1}{4}$", ha="center", fontsize=6.5,
            color=MERAH)
    ax.text(2, -0.78, r"aksi kanan di keadaan 2: ke 3 dengan peluang "
            r"$\frac{3}{4}$, ke 1 dengan peluang $\frac{1}{4}$; "
            r"$\gamma = \frac{1}{2}$", ha="center", fontsize=5.8)
    simpan(fig, "bab01-lorong")


def bab01_konvergensi():
    from bab01_lorong import (ACAK, BENIH, KANAN_SEMUA, OPTIMUM, Lorong,
                              jalankan_episode, nilai_eksak, return_)
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    n = 3000
    fig, ax = plt.subplots(figsize=(4.3, 2.0))
    for nama, pi, w in (("acak seragam", ACAK, ABU),
                        ("selalu kanan", KANAN_SEMUA, BIRU),
                        ("kiri di 1, kanan lainnya", OPTIMUM, HIJAU)):
        g = np.array([return_([r for _, _, r in
                               jalankan_episode(env, pi, rng)])[0]
                      for _ in range(n)])
        rata = np.cumsum(g) / np.arange(1, n + 1)
        ax.plot(np.arange(1, n + 1), rata, color=w, lw=0.8, label=nama)
        ax.axhline(nilai_eksak(pi)[1], color=w, lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("banyaknya episode")
    ax.set_ylabel(r"rata-rata $G_0$ dari keadaan 2")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=5.8, loc="upper right", ncol=1)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-konvergensi")


def bab01_gamma():
    g = np.linspace(0.05, 0.99, 300)
    v_kanan = 6 * g ** 2 / (1 - g ** 2 / 4)
    g0 = 2 / np.sqrt(13)
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.plot(g, np.full_like(g, 2.0), color=JINGGA, lw=0.9,
            label="kiri di 1: $2$")
    ax.plot(g, v_kanan, color=BIRU, lw=0.9,
            label=r"kanan di 1: $6\gamma^2/(1 - \gamma^2/4)$")
    ax.axvline(g0, color=ABU_GARIS, lw=0.7, ls="--")
    ax.text(g0 + 0.015, 5.2, r"$\gamma = 2/\sqrt{13}$", fontsize=6)
    ax.plot([0.5, 0.9], [2, 2], "o", ms=3, color=JINGGA)
    ax.plot([0.5, 0.9], [6 * 0.25 / (1 - 0.0625), 6 * 0.81 / (1 - 0.2025)],
            "o", ms=3, color=BIRU)
    ax.set_xlabel(r"faktor diskon $\gamma$")
    ax.set_ylabel("nilai keadaan 1")
    ax.set_ylim(0, 7)
    ax.legend(fontsize=5.8, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-gamma")


# ============================ Bab 2 ==================================

def bab02_regret():
    from bab02_bandit import MU_MINI, STRATEGI, simulasi
    warna = [ABU, BIRU, "#6C8EBF", JINGGA, HIJAU, MERAH]
    fig, ax = plt.subplots(figsize=(4.3, 2.2))
    t = np.arange(1, 1001)
    for (nama, s, kw), w in zip(STRATEGI, warna):
        reg, _ = simulasi(MU_MINI, s, **kw)
        lbl = nama.replace("eps", r"$\varepsilon$").replace(
            "Q0=1", "$Q_0 = 1$")
        ax.plot(t, reg, color=w, lw=0.9, label=lbl)
    ax.set_xlabel("langkah $t$")
    ax.set_ylabel("regret kumulatif")
    ax.legend(fontsize=5.6, loc="upper left", ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-regret")


def bab02_beta():
    from scipy.stats import beta
    x = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    ax.plot(x, beta.pdf(x, 2, 2), color=BIRU, lw=0.9,
            label="lengan 1 dan 2: Beta$(2, 2)$")
    ax.plot(x, beta.pdf(x, 4, 2), color=HIJAU, lw=0.9,
            label="lengan 3: Beta$(4, 2)$")
    for m, w in ((0.25, BIRU), (0.5, BIRU), (0.75, HIJAU)):
        ax.axvline(m, color=w, lw=0.5, ls=":")
    ax.set_xlabel(r"peluang sukses $\theta$")
    ax.set_ylabel("kepadatan posterior")
    ax.legend(fontsize=5.8, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-beta")


def bab02_bobot():
    n = 20
    i = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    ax.plot(i, np.full(n, 1 / n), "o-", color=ABU, ms=2.5, lw=0.7,
            label=r"rata-rata sampel, $1/n$")
    for a, w in ((0.1, BIRU), (0.3, JINGGA)):
        ax.plot(i, a * (1 - a) ** (n - i), "o-", color=w, ms=2.5, lw=0.7,
                label=rf"$\alpha = {angka_mat(a, 1)}$")
    ax.set_xlabel("urutan imbalan $i$ (yang terbaru $i = 20$)")
    ax.set_ylabel(r"bobot imbalan $R_i$")
    ax.legend(fontsize=5.8, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-bobot")


# ============================ Bab 3 ==================================

def bab03_sebaran():
    from bab03_mdp import acak, mdp_lorong
    m = mdp_lorong()
    Ppi, _ = m.rantai(acak(m))
    d = np.zeros(5)
    d[2] = 1.0
    D = []
    for _ in range(11):
        D.append(d)
        d = d @ Ppi
    D = np.array(D)
    t = np.arange(11)
    fig, ax = plt.subplots(figsize=(4.3, 1.95))
    bawah = np.zeros(11)
    for kol, nama, w in ((0, "keluar kiri (0)", JINGGA),
                         (1, "petak 1", "#9DB4D6"),
                         (2, "petak 2", MERAH),
                         (3, "petak 3", BIRU),
                         (4, "keluar kanan (4)", HIJAU)):
        ax.bar(t, D[:, kol], bottom=bawah, color=w, width=0.75,
               label=nama, edgecolor="white", lw=0.3)
        bawah += D[:, kol]
    ax.set_xlabel("langkah $t$")
    ax.set_ylabel(r"$\mathbb{P}(S_t = s)$")
    ax.set_xticks(t)
    ax.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-sebaran")


def bab03_deret():
    from bab03_mdp import acak, mdp_lorong
    m = mdp_lorong()
    Ppi, rpi = m.rantai(acak(m))
    Pt = Ppi[1:4, 1:4]
    v = m.nilai(acak(m))[1:4]
    K = np.arange(1, 41)
    jumlah, suku, gal = np.zeros(3), rpi[1:4].copy(), []
    for _ in K:
        jumlah += suku
        suku = m.gamma * Pt @ suku
        gal.append(np.abs(jumlah - v).max())
    rho = m.gamma * np.abs(np.linalg.eigvals(Pt)).max()
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    ax.semilogy(K, gal, "o", ms=2.2, color=BIRU, label="galat jumlah parsial")
    ax.semilogy(K, gal[0] * rho ** (K - 1), color=JINGGA, lw=0.8,
                label=r"$(\gamma\rho)^{K}$, $\gamma\rho = 0{,}3536$")
    ax.set_xlabel("banyaknya suku $K$")
    ax.set_ylabel("galat maksimum")
    ax.legend(fontsize=5.8)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-deret")


# ============================ Bab 4 ==================================

def bab04_cadangan():
    from matplotlib.patches import Circle
    fig, axs = plt.subplots(1, 4, figsize=(4.4, 1.35))
    judul = [r"$v_\pi$", r"$q_\pi$", r"$v_*$", r"$q_*$"]

    def keadaan(ax, x, y):
        ax.add_patch(Circle((x, y), 0.11, facecolor="white",
                            edgecolor=BIRU, lw=0.8, zorder=3))

    def aksi(ax, x, y):
        ax.plot(x, y, "o", ms=3.4, color=BIRU, zorder=3)

    def garis(ax, a, b):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ABU, lw=0.6, zorder=1)

    def busur(ax, x, y, lebar):
        from matplotlib.patches import Arc
        ax.add_patch(Arc((x, y), lebar, 0.35, theta1=200, theta2=340,
                         color=JINGGA, lw=0.9))

    for i, ax in enumerate(axs):
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.35, 0.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(judul[i], fontsize=8, pad=2)
        mulai_v = i in (0, 2)
        if mulai_v:
            keadaan(ax, 0, 0.2)
            xs_a = (-0.5, 0.5)
            for xa in xs_a:
                garis(ax, (0, 0.2), (xa, -0.45))
                aksi(ax, xa, -0.45)
                for xs in (xa - 0.25, xa + 0.25):
                    garis(ax, (xa, -0.45), (xs, -1.1))
                    keadaan(ax, xs, -1.1)
            if i == 2:
                busur(ax, 0, 0.05, 0.9)
        else:
            aksi(ax, 0, 0.2)
            for xs in (-0.5, 0.5):
                garis(ax, (0, 0.2), (xs, -0.45))
                keadaan(ax, xs, -0.45)
                for xa in (xs - 0.25, xs + 0.25):
                    garis(ax, (xs, -0.45), (xa, -1.1))
                    aksi(ax, xa, -1.1)
                if i == 3:
                    busur(ax, xs, -0.62, 0.55)
    fig.text(0.5, 0.06, r"lingkaran: keadaan $s$, titik: pasangan $(s, a)$,"
             r" busur: maksimum atas aksi", ha="center", fontsize=5.8)
    simpan(fig, "bab04-cadangan")


def bab04_kontraksi():
    from bab03_mdp import mdp_lorong
    from bab04_bellman import T_opt
    m = mdp_lorong(gamma=0.9)
    v_opt = m.nilai(np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 0.]]))
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    k = np.arange(0, 61)
    for v0, nama, w in ((np.zeros(5), "$v_0 = 0$", BIRU),
                        (np.array([0, 100, 100, 100, 0.]), "$v_0 = 100$",
                         HIJAU),
                        (np.array([0, -50, 80, -20, 0.]), "$v_0$ acak",
                         JINGGA)):
        v, gal = v0.copy(), []
        for _ in k:
            gal.append(np.abs(v - v_opt).max())
            v = T_opt(m, v)
        gal = np.array(gal)
        ax.semilogy(k, np.maximum(gal, 1e-17), "o-", ms=2, lw=0.6,
                    color=w, label=nama)
        ax.semilogy(k, gal[0] * 0.9 ** k, color=w, lw=0.5, ls=":")
    ax.set_xlabel("iterasi $k$")
    ax.set_ylabel(r"$\|v_k - v_*\|_\infty$")
    ax.set_ylim(1e-16, 1e3)
    ax.legend(fontsize=5.8, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-kontraksi")


# ============================ Bab 5 ==================================

def bab05_grid():
    from bab05_dp import PETA_GRID, iterasi_kebijakan, mdp_grid
    mg = mdp_grid(gamma=0.99)
    pi, v, _ = iterasi_kebijakan(mg)
    n = len(PETA_GRID)
    V = v.reshape(n, n)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
    panah = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
    for i in range(n):
        for j in range(n):
            c = PETA_GRID[i][j]
            if c == "L":
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=MERAH_MUDA,
                                           edgecolor="white"))
                ax.text(j, i, "L", ha="center", va="center", fontsize=6,
                        color=MERAH)
                continue
            if c == "G":
                ax.text(j, i, "G", ha="center", va="center", fontsize=7,
                        color="white", weight="bold")
                continue
            dx, dy = panah[int(pi[i * n + j].argmax())]
            w = "white" if V[i, j] > 0.6 else BIRU
            ax.annotate("", xy=(j + dx, i + dy), xytext=(j - dx, i - dy),
                        arrowprops=dict(arrowstyle="-|>", color=w, lw=0.6,
                                        mutation_scale=5))
            ax.text(j, i + 0.36, angka(V[i, j], 2), ha="center",
                    va="center", fontsize=3.6, color=w)
    ax.set_xticks([])
    ax.set_yticks([])
    simpan(fig, "bab05-grid")


def bab05_residu():
    from bab04_bellman import T_opt
    from bab05_dp import iterasi_kebijakan, iterasi_nilai, mdp_grid
    mg = mdp_grid(gamma=0.99)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    for nama, dt, w in (("VI serentak", False, BIRU),
                        ("VI di tempat", True, JINGGA)):
        _, _, res = iterasi_nilai(mg, theta=1e-13, ditempat=dt)
        ax.semilogy(np.arange(1, len(res) + 1), res, color=w, lw=0.9,
                    label=nama)
    _, _, riw = iterasi_kebijakan(mg)
    res = [np.abs(T_opt(mg, v) - v).max() for _, v in riw]
    res = np.maximum(res, 1e-16)
    ax.semilogy(np.arange(1, len(res) + 1), res, "o-", color=HIJAU,
                ms=2.5, lw=0.8, label="policy iteration")
    ax.set_xlabel("iterasi (sapuan)")
    ax.set_ylabel(r"residu $\|T^*v - v\|_\infty$")
    ax.set_ylim(1e-16, 2)
    ax.legend(fontsize=5.8)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-residu")


# ============================ Bab 6 ==================================

def bab06_rmse():
    from bab01_lorong import ACAK, BENIH, OPTIMUM, Lorong, jalankan_episode
    from bab06_mc import mc_prediksi
    from bab06_offpolicy import is_estimasi
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    ulang, nmaks = 500, 1000
    data = [[jalankan_episode(env, ACAK, rng) for _ in range(nmaks)]
            for _ in range(ulang)]
    ns = np.unique(np.logspace(0, 3, 13).astype(int))
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 1.9), sharey=True)
    for pertama, nama, w in ((True, "first-visit", BIRU),
                             (False, "every-visit", JINGGA)):
        r = [np.sqrt(np.mean([(mc_prediksi(d[:n], pertama=pertama)[0][2]
                               - 10 / 7) ** 2 for d in data])) for n in ns]
        axs[0].loglog(ns, r, "o-", ms=2, lw=0.8, color=w, label=nama)
    axs[0].set_title(r"prediksi $V(2)$, kebijakan acak", fontsize=6.5)
    for bb, nama, w in ((False, "IS biasa", MERAH),
                        (True, "IS berbobot", HIJAU)):
        r = [np.sqrt(np.mean([(is_estimasi(d[:n], OPTIMUM, ACAK,
                                           berbobot=bb) - 3.25) ** 2
                              for d in data])) for n in ns]
        axs[1].loglog(ns, r, "o-", ms=2, lw=0.8, color=w, label=nama)
    axs[1].set_title(r"off-policy $V(2)$ untuk -++", fontsize=6.5)
    for ax in axs:
        ax.loglog(ns, 1.5 / np.sqrt(ns), color=ABU_GARIS, lw=0.6, ls="--")
        ax.set_xlabel("banyaknya episode $n$")
        ax.legend(fontsize=5.3)
        _rapikan(ax)
    axs[0].set_ylabel("RMSE")
    fig.tight_layout()
    simpan(fig, "bab06-rmse")


def bab06_kontrol():
    from bab05_dp import iterasi_kebijakan, mdp_grid
    from bab06_mc import mc_kontrol
    mg = mdp_grid(gamma=0.99)
    _, v_opt, _ = iterasi_kebijakan(mg)
    catat = sorted(set(np.logspace(2, np.log10(20000), 12).astype(int)))
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    for eps, q0, w, nama in ((0.05, 0.0, ABU, r"$\varepsilon = 0{,}05$, $Q_0 = 0$"),
                             (0.05, 1.0, BIRU, r"$\varepsilon = 0{,}05$, $Q_0 = 1$"),
                             (0.1, 1.0, HIJAU, r"$\varepsilon = 0{,}1$, $Q_0 = 1$")):
        _, kurva = mc_kontrol(mg, max(catat), eps=eps, catat=set(catat),
                              q0=q0)
        ax.semilogx(catat, kurva, "o-", ms=2, lw=0.8, color=w, label=nama)
    ax.axhline(v_opt[0], color=ABU, lw=0.6, ls="--")
    ax.text(110, v_opt[0] + 0.02, r"$V_*(\mathrm{awal})$", fontsize=6)
    ax.set_xlabel("banyaknya episode")
    ax.set_ylabel("$V$(awal), kebijakan greedy")
    ax.set_ylim(-0.03, 0.7)
    ax.legend(fontsize=5.4, loc="center right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-kontrol")

# ============================ Bab 7 ==================================

def bab07_rmse():
    from bab07_td import kurva_rmse
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    ep = np.arange(1, 101)
    for metode, a, w, ls in (("td", 0.05, BIRU, "-"), ("td", 0.1, HIJAU, "-"),
                             ("mc", 0.05, BIRU, "--"),
                             ("mc", 0.1, HIJAU, "--")):
        k = kurva_rmse(metode, a)
        ax.plot(ep, k, color=w, ls=ls, lw=0.9,
                label=f"{metode.upper()}, $\\alpha = {angka_mat(a, 2)}$")
    ax.set_xlabel("episode")
    ax.set_ylabel("RMSE $V$(1..3)")
    ax.legend(fontsize=5.6, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-rmse")

# ============================ Bab 8 ==================================

def bab08_tebing():
    from bab08_qlearning import BENIH, Tebing, latih
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    k = 10
    for nama, met, w in (("SARSA", "sarsa", BIRU), ("Q-learning", "q", MERAH),
                         ("expected SARSA", "esarsa", HIJAU)):
        kurva = np.mean([latih(Tebing(), met, 500, benih=BENIH + u)[1]
                         for u in range(50)], axis=0)
        halus = np.convolve(kurva, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k, 501), halus, color=w, lw=0.9, label=nama)
    ax.set_ylim(-120, 0)
    ax.set_xlabel("episode")
    ax.set_ylabel("jumlah imbalan per episode")
    ax.legend(fontsize=5.8, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-tebing")


def bab08_jalur():
    from bab08_qlearning import BENIH, Tebing, jalur_greedy, latih
    fig, ax = plt.subplots(figsize=(4.2, 1.6))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_aspect("equal")
    for j in range(1, 11):
        ax.add_patch(plt.Rectangle((j - .5, 2.5), 1, 1, facecolor=MERAH_MUDA,
                                   edgecolor="white"))
    ax.text(5.5, 3, "tebing ($-100$)", ha="center", va="center",
            fontsize=6, color=MERAH)
    ax.text(0, 3, "A", ha="center", va="center", fontsize=7, color=BIRU)
    ax.text(11, 3, "T", ha="center", va="center", fontsize=7, color=BIRU)
    for nama, met, w, geser in (("SARSA", "sarsa", BIRU, -0.08),
                                ("Q-learning", "q", MERAH, 0.08)):
        Q, _ = latih(Tebing(), met, 500, benih=BENIH)
        jalur = jalur_greedy(Q, Tebing())
        ij = np.array([divmod(s, 12) for s in jalur], float)
        ax.plot(ij[:, 1], ij[:, 0] + geser, color=w, lw=1.1, label=nama)
    for x in np.arange(-0.5, 12, 1):
        ax.axvline(x, color=ABU_GARIS, lw=0.3)
    for y in np.arange(-0.5, 4, 1):
        ax.axhline(y, color=ABU_GARIS, lw=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=5.8, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.28))
    simpan(fig, "bab08-jalur")


def bab08_bias():
    from bab08_ganda import jalankan
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    e = np.arange(1, 301)
    ax.plot(e, jalankan(False), color=MERAH, lw=0.9, label="Q-learning")
    ax.plot(e, jalankan(True), color=HIJAU, lw=0.9, label="double Q-learning")
    ax.axhline(0.05, color=ABU, lw=0.6, ls="--")
    ax.text(250, 0.08, "optimal", fontsize=6)
    ax.set_xlabel("episode")
    ax.set_ylabel("bagian memilih kiri di A")
    ax.legend(fontsize=5.8)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-bias")

# ============================ Bab 9 ==================================

def bab09_alpha():
    from bab09_jejak import galat_rata
    alphas = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    warna = [ABU, "#6C8EBF", BIRU, HIJAU, JINGGA]
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.0), sharey=True)
    for ax, metode, pars, judul in ((axs[0], "n", (1, 2, 4, 8, 16),
                                     "TD $n$-langkah"),
                                    (axs[1], "l", (0.0, 0.4, 0.8, 0.9, 1.0),
                                     r"TD($\lambda$) daring")):
        for p, w in zip(pars, warna):
            with np.errstate(all="ignore"):
                r = np.array([galat_rata(metode, p, a) for a in alphas])
            r[~(r < 0.62)] = np.nan
            lbl = f"$n = {p}$" if metode == "n" else \
                rf"$\lambda = {angka_mat(p, 1)}$"
            ax.plot(alphas, r, "o-", ms=2, lw=0.8, color=w, label=lbl)
        ax.set_title(judul, fontsize=6.8)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylim(0.24, 0.72)
        ax.legend(fontsize=4.8, loc="upper center", ncol=3,
                  columnspacing=0.6, handlelength=1.2)
        _rapikan(ax)
    axs[0].set_ylabel("RMS rata-rata 10 episode")
    fig.tight_layout()
    simpan(fig, "bab09-alpha")

# ============================ Bab 10 =================================

def bab10_dyna():
    from bab10_perencanaan import BENIH, Labirin, PETA, dyna_q
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 1.9),
                            gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axs[0]
    n, m = len(PETA), len(PETA[0])
    for i in range(n):
        for j in range(m):
            c = PETA[i][j]
            isi = {"X": ABU, "S": BIRU_MUDA, "G": HIJAU_MUDA}.get(c, "white")
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=isi,
                                       edgecolor=ABU_GARIS, lw=0.4))
            if c in "SG":
                ax.text(j + .5, n - .5 - i, c, ha="center", va="center",
                        fontsize=6, color=BIRU)
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = axs[1]
    for k, w in ((0, ABU), (5, BIRU), (50, HIJAU)):
        L = np.mean([dyna_q(Labirin(), k, benih=BENIH + u)[1]
                     for u in range(30)], axis=0)
        ax.semilogy(np.arange(2, 51), L[1:], color=w, lw=0.9,
                    label=f"$n = {k}$")
    ax.set_xlabel("episode")
    ax.set_ylabel("langkah per episode")
    ax.legend(fontsize=5.6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-dyna")


def bab10_uct():
    from bab10_perencanaan import uct
    ns = np.unique(np.logspace(1, 4, 13).astype(int))
    hasil = np.array([uct(2, n)[0] for n in ns])
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    ax.semilogx(ns, hasil[:, 1], "o-", ms=2, color=BIRU, lw=0.8,
                label="rata-rata akar, kanan")
    ax.semilogx(ns, hasil[:, 0], "o-", ms=2, color=JINGGA, lw=0.8,
                label="rata-rata akar, kiri")
    ax.axhline(3.25, color=BIRU, lw=0.5, ls="--")
    ax.axhline(1.75, color=JINGGA, lw=0.5, ls="--")
    ax.set_xlabel("banyaknya simulasi")
    ax.set_ylabel("taksiran $Q(2, \\cdot)$")
    ax.legend(fontsize=5.6, loc="center right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-uct")

# ============================ Bab 11 =================================

def bab11_fitur():
    from bab11_linear import (BENIH, JalanAcak1000, fitur_agregasi,
                              fitur_fourier, gradien_mc, lstd, semi_td)
    jl = JalanAcak1000()
    v, _ = jl.nilai_benar()
    rng = np.random.default_rng(BENIH)
    data = [jl.episode(rng) for _ in range(2000)]
    s = np.arange(1, 1001)
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 1.9), sharey=True)
    for ax, f, d, a_mc, a_td, judul in (
            (axs[0], fitur_agregasi, 10, 3e-4, 0.1, "agregasi 10 kelompok"),
            (axs[1], fitur_fourier, 6, 3e-5, 3e-3, "Fourier orde 5")):
        ax.plot(s, v, color=ABU, lw=1.6, label="nilai benar")
        for metode, a, w_, nama in ((gradien_mc, a_mc, JINGGA, "gradien MC"),
                                    (semi_td, a_td, BIRU, "semi-gradien TD")):
            w = np.zeros(d)
            for tr in data:
                metode(w, tr, f, a)
            ax.plot(s, [w @ f(x) for x in s], color=w_, lw=0.8, label=nama)
        ax.set_title(judul, fontsize=6.8)
        ax.set_xlabel("petak")
        _rapikan(ax)
    axs[0].set_ylabel("nilai")
    axs[0].legend(fontsize=5.2, loc="upper left")
    fig.tight_layout()
    simpan(fig, "bab11-fitur")


def bab11_baird():
    from bab11_linear import baird
    _, j = baird(2000)
    fig, ax = plt.subplots(figsize=(3.8, 1.7))
    ax.semilogy(np.arange(len(j)), j, color=MERAH, lw=0.9)
    ax.set_xlabel("sapuan")
    ax.set_ylabel(r"$\|\mathbf{w}\|$")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-baird")

# ============================ Bab 12 =================================

def bab12_kurva():
    from bab12_hasil import DATA, VARIAN
    h = np.load(DATA)
    warna = {"dqn": BIRU, "double": HIJAU, "tanpa_target": MERAH,
             "tanpa_replay": JINGGA}
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for kunci, nama in VARIAN:
        ev = np.array([h[f"{kunci}_{u}_evalk"] for u in range(3)])
        t = np.arange(1, ev.shape[1] + 1) * 5000
        ax.plot(t, ev.mean(0), "o-", ms=2, lw=0.9, color=warna[kunci],
                label=nama)
        ax.fill_between(t, ev.min(0), ev.max(0), color=warna[kunci],
                        alpha=0.12, lw=0)
    ax.set_xlabel("langkah lingkungan")
    ax.set_ylabel("return greedy")
    ax.set_ylim(0, 520)
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-kurva")

# ============================ Bab 13 =================================

def bab13_lorong():
    from bab13_reinforce import BENIH, latih
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    for bl, w, nama in ((False, JINGGA, "tanpa baseline"),
                        (True, BIRU, "baseline")):
        k = np.mean([latih(1000, 0.5, BENIH + u, baseline=bl)[0]
                     for u in range(20)], 0)
        ax.semilogx(np.arange(1, 1001), k, color=w, lw=0.9, label=nama)
    ax.axhline(3.25, color=ABU, lw=0.6, ls="--")
    ax.set_xlabel("episode")
    ax.set_ylabel(r"$J(\theta)$")
    ax.legend(fontsize=5.8, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-lorong")


def bab13_cartpole():
    from bab13_hasil import DATA
    h = np.load(DATA)
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    for kunci, w, nama in (("tanpa", JINGGA, "tanpa baseline"),
                           ("baseline", BIRU, "baseline $\\hat V$")):
        k = np.array([h[f"{kunci}_{u}"] for u in range(3)])
        halus = np.array([np.convolve(x, np.ones(20) / 20, "valid") for x in k])
        e = np.arange(20, 1001)
        ax.plot(e, halus.mean(0), color=w, lw=0.9, label=nama)
        ax.fill_between(e, halus.min(0), halus.max(0), color=w, alpha=0.12,
                        lw=0)
    ax.set_xlabel("episode")
    ax.set_ylabel("panjang episode")
    ax.set_ylim(0, 520)
    ax.legend(fontsize=5.8, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-cartpole")

# ============================ Bab 14 =================================

def bab14_a2c():
    from bab14_hasil import DATA, VARIAN
    h = np.load(DATA)
    warna = {"lam0": BIRU, "lam095": HIJAU, "lam1": JINGGA, "berbagi": MERAH}
    nama = {"lam0": r"$\lambda = 0$", "lam095": r"$\lambda = 0{,}95$",
            "lam1": r"$\lambda = 1$", "berbagi": r"berbagi, $\lambda = 0{,}95$"}
    grid = np.arange(0, 300_001, 10_000)
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for kunci, _ in VARIAN:
        kurva = []
        for u in range(3):
            k = h[f"{kunci}_{u}"]
            kurva.append([k[(k[:, 0] >= a) & (k[:, 0] < a + 10_000), 1].mean()
                          for a in grid[:-1]])
        kurva = np.array(kurva)
        x = grid[:-1] + 5000
        ax.plot(x, kurva.mean(0), color=warna[kunci], lw=0.9,
                label=nama[kunci])
        ax.fill_between(x, kurva.min(0), kurva.max(0), color=warna[kunci],
                        alpha=0.1, lw=0)
    ax.set_xlabel("langkah lingkungan")
    ax.set_ylabel("return episode")
    ax.set_ylim(0, 520)
    ax.legend(fontsize=5.4, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-a2c")

# ============================ Bab 15 =================================

def bab15_natural():
    from bab03_mdp import mdp_lorong
    from bab15_natural import lintasan
    m = mdp_lorong()
    th0 = np.zeros((5, 2))
    th0[:, 0] = np.log(9)
    k = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    ax.semilogx(k, lintasan(th0, 200, 1.0, False, m), color=JINGGA, lw=0.9,
                label=r"gradien biasa, $\alpha = 1$")
    ax.semilogx(k, lintasan(th0, 200, 0.5, True, m), color=BIRU, lw=0.9,
                label=r"gradien natural, $\alpha = 0{,}5$")
    ax.axhline(3.25, color=ABU, lw=0.6, ls="--")
    ax.set_xlabel("langkah")
    ax.set_ylabel(r"$J(\theta)$")
    ax.legend(fontsize=5.8, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-natural")


def bab15_ppo():
    from bab15_hasil import DATA
    h = np.load(DATA)
    grid = np.arange(0, 300_001, 10_000)
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 1.9), sharey=True)
    for ax, konfig, judul in ((axs[0], "std", "standar"),
                              (axs[1], "agresif", "agresif")):
        for varian, w, nama in (("clip", BIRU, "PPO-clip"),
                                ("tanpa_clip", MERAH, "tanpa clipping")):
            kurva = []
            for u in range(3):
                k = h[f"{konfig}_{varian}_{u}"]
                kurva.append([k[(k[:, 0] >= a) & (k[:, 0] < a + 10_000),
                                1].mean() for a in grid[:-1]])
            kurva = np.array(kurva)
            x = grid[:-1] + 5000
            ax.plot(x, kurva.mean(0), color=w, lw=0.9, label=nama)
            ax.fill_between(x, kurva.min(0), kurva.max(0), color=w,
                            alpha=0.12, lw=0)
        ax.set_title(judul, fontsize=6.8)
        ax.set_xlabel("langkah")
        ax.set_ylim(0, 520)
        _rapikan(ax)
    axs[0].set_ylabel("return episode")
    axs[0].legend(fontsize=5.4, loc="lower right")
    fig.tight_layout()
    simpan(fig, "bab15-ppo")

# ============================ Bab 17 =================================

def bab17_berlebihan():
    from bab17_rlhf import (BENIH, FITUR, PI_REF, R_BENAR, data_preferensi,
                            kl, latih_model_imbalan, optimum_kl)
    rng = np.random.default_rng(BENIH)
    menang, kalah = data_preferensi(1000, rng)
    r_hat = FITUR @ latih_model_imbalan(menang, kalah)
    betas = np.logspace(1, -1.5, 40)
    K, Rh, Rb, Rs, Ks = [], [], [], [], []
    for b in betas:
        p = optimum_kl(PI_REF, r_hat, b)
        K.append(kl(p, PI_REF)); Rh.append(p @ r_hat); Rb.append(p @ R_BENAR)
        q = optimum_kl(PI_REF, R_BENAR, b)
        Ks.append(kl(q, PI_REF)); Rs.append(q @ R_BENAR)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    ax.plot(K, Rh, color=JINGGA, lw=0.9, label=r"imbalan model $\hat r$")
    ax.plot(K, Rb, color=MERAH, lw=0.9, label=r"imbalan sebenarnya $r^*$")
    ax.plot(Ks, Rs, color=HIJAU, lw=0.9, ls="--",
            label=r"optimum dengan $r^*$")
    ax.set_xlabel(r"KL$(\pi \,\|\, \pi_{\mathrm{ref}})$")
    ax.set_ylabel("harapan imbalan")
    ax.legend(fontsize=5.6, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-berlebihan")

# ============================ Bab 16 =================================

def bab16_kurva():
    from bab16_hasil import DATA
    h = np.load(DATA)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    t = np.arange(1, 16) * 1000
    for algo, w, nama in (("ddpg", JINGGA, "DDPG"), ("td3", BIRU, "TD3"),
                          ("sac", HIJAU, "SAC")):
        k = np.array([h[f"{algo}_{u}"] for u in range(3)])
        ax.plot(t, k.mean(0), "o-", ms=2, lw=0.9, color=w, label=nama)
        ax.fill_between(t, k.min(0), k.max(0), color=w, alpha=0.12, lw=0)
    ax.axhline(-1241, color=ABU, lw=0.6, ls="--")
    ax.text(9000, -1180, "kebijakan acak", fontsize=5.8)
    ax.set_xlabel("langkah lingkungan")
    ax.set_ylabel("return greedy")
    ax.legend(fontsize=5.6, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-kurva")

# ============================ Bab 18 =================================

def bab18_sebaran():
    from bab13_reinforce import latih
    from bab18_riset import BENIH
    a = np.array([latih(50, 0.5, BENIH + u)[0][-1] for u in range(40)])
    b = np.array([latih(50, 0.5, BENIH + u, baseline=True)[0][-1]
                  for u in range(40)])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.0, 1.7))
    for i, (v, w, nama) in enumerate(((a, JINGGA, "tanpa baseline"),
                                      (b, BIRU, "baseline"))):
        ax.plot(v, i + rng.uniform(-0.15, 0.15, len(v)), "o", ms=2.6,
                color=w, alpha=0.7)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=w, lw=1.2)
    ax.axvline(3.25, color=ABU, lw=0.6, ls="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["tanpa baseline", "baseline"])
    kunci_label(ax, "y")
    ax.set_xlabel(r"$J(\theta)$ sesudah 50 episode")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-sebaran")


if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
