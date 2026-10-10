# Kode — Memahami Reinforcement Learning

Repositori pendamping buku **_Memahami Reinforcement Learning: Dari
Persamaan Bellman sampai PPO, Episode demi Episode_** (Edisi Pertama,
2026) oleh Mohammad Jamhuri, seri *Memahami*. Tag `edisi-1` menandai
kode yang tepat dipakai untuk mencetak edisi pertama.

Buku ini memakai satu **data mini** dari awal sampai akhir: *lorong
licin*, yaitu petak 0 sampai 4 dengan pintu keluar di petak 0 (imbalan
+2) dan petak 4 (imbalan +8), agen mulai di petak 2, aksi kiri atau
kanan, petak 2 licin (berhasil dengan peluang 3/4), dan faktor diskon
1/2. Kebijakan optimalnya kiri di petak 1 dan kanan di petak 2 dan 3,
dengan V* = (2, 13/4, 8). File `kode/babNN_contoh.py` memeriksa setiap
bilangan di kotak Contoh Soal Bab NN.

**Setiap angka keluaran yang tercetak di buku dihasilkan oleh kode di
sini**, dan `periksa.py` membuktikannya: skrip itu menjalankan ulang
kode setiap bab dan mencocokkan hasilnya dengan blok keluaran yang
tercetak di buku (`keluaran/babNN.txt`).

Semua lingkungan ditulis sendiri dengan antarmuka gaya Gymnasium
(`reset()`, `step(a)`): lorong licin (`kode/bab01_lorong.py`), grid es
8 x 8 (`kode/bab05_dp.py`), tebing (`kode/bab08_qlearning.py`), labirin
(`kode/bab10_perencanaan.py`), CartPole (`kode/bab12_cartpole.py`), dan
Pendulum (`kode/bab16_pendulum.py`). Tidak ada paket lingkungan yang
perlu dipasang. Versi minimal satu berkas ada di `kode/lampiran_rl.py`.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)). Semua data dibangkitkan
oleh kode ini sendiri.

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-reinforcement-learning.git
cd buku-memahami-reinforcement-learning
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab01_lorong.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
memakai benih tetap 20261009 untuk agen dan 20261010 untuk lingkungan
(Jebakan Umum Bab 1). Semua kode berjalan di CPU.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab (beberapa menit di CPU)
.venv/bin/python periksa.py 08     # Bab 8 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. GitHub Actions menjalankan pemeriksaan ini beserta
semua Contoh Soal pada setiap perubahan.

## Pelatihan jaringan saraf (Bab 12–16)

Bab 12–16 melatih jaringan dengan PyTorch. Hasil pelatihan disimpan di
`data/`, dan skrip `kode/babNN_hasil.py` mencetak keluaran buku dari
berkas itu. Untuk melatih ulang (waktu di satu inti CPU):

| Bab | Skrip | Berkas hasil | Waktu |
|---|---|---|---|
| 12 | `kode/bab12_dqn.py` | `data/bab12_dqn.npz` | ±20 menit |
| 13 | `kode/bab13_cartpole.py` | `data/bab13_reinforce.npz` | ±3 menit |
| 14 | `kode/bab14_a2c.py` | `data/bab14_a2c.npz` | ±4 menit |
| 15 | `kode/bab15_ppo.py` | `data/bab15_ppo.npz` | ±8 menit |
| 16 | `kode/bab16_kontinu.py` | `data/bab16_kontinu.npz` | ±15 menit |

Pada CPU atau versi pustaka yang berbeda, angka pelatihan dapat berbeda
di digit terakhir atau lebih, tetapi kesimpulannya sama.

## Gambar

`python gen_gambar.py` membangkitkan semua gambar buku ke folder `gbr/`
(sebagian menjalankan simulasi dan memerlukan beberapa menit).
