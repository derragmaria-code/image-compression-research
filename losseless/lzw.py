"""
╔══════════════════════════════════════════════════════════════╗
║   LZW IMAGE COMPRESSION — KAGGLE OPTIMIZED (v3 FIXED)       ║
║   ✔ Vraie compression LZW (encode/decode bitstream)         ║
║   ✔ Reconstruction image réelle                              ║
║   ✔ Table Pandas 800 images + export CSV                    ║
║   ✔ Graphes propres (PSNR, SSIM, BPP)                       ║
║   ✔ Affichage original vs reconstructed                     ║
╚══════════════════════════════════════════════════════════════╝

Fix v3 — règle d'élargissement asymétrique (seule combinaison correcte) :
  ENCODEUR : élargit cw APRÈS avoir ajouté nc, quand nc > 2^cw
             → le prochain code ÉMIS utilise le nouveau cw
  DÉCODEUR : élargit cw AVANT de lire le prochain code, quand nc == 2^cw
             → compense le décalage inhérent (le décodeur ajoute
               les entrées une itération après l'encodeur)

  Codes spéciaux : CLEAR=256, STOP=257, premier code libre=258.
  Codes sur 9 à 12 bits (dict max 4096 entrées), reset auto.
  Bitstream MSB-first avec header 4 octets = nb de bits utiles.
"""

import os
import glob
import struct
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
KAGGLE_PATH   = "/kaggle/input/datasets/mariyyyaaella/div2k/DIV2K_train_HR"
FALLBACK_PATH = "./DIV2K_train_HR"
MAX_IMAGES    = 800
CROP_SIZE     = 256
OUTPUT_DIR    = "./lzw_output"
CSV_NAME      = "lzw_metrics.csv"

MIN_BITS = 9
MAX_BITS = 12
CLEAR    = 256
STOP     = 257
FIRST    = 258

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. BIT-PACKER / UNPACKER (MSB-first, variable width)
# ──────────────────────────────────────────────────────────────

class BitWriter:
    def __init__(self):
        self._buf = 0
        self._n   = 0
        self._out = bytearray()

    def write(self, code: int, width: int):
        self._buf  = (self._buf << width) | (code & ((1 << width) - 1))
        self._n   += width
        while self._n >= 8:
            self._n -= 8
            self._out.append((self._buf >> self._n) & 0xFF)

    def flush(self) -> bytes:
        remainder = 0
        if self._n > 0:
            self._out.append((self._buf << (8 - self._n)) & 0xFF)
            remainder = 8 - self._n
        useful = len(self._out) * 8 - remainder
        return struct.pack(">I", useful) + bytes(self._out)


class BitReader:
    def __init__(self, data: bytes):
        useful     = struct.unpack(">I", data[:4])[0]
        self._bits = "".join(f"{b:08b}" for b in data[4:])[:useful]
        self._pos  = 0

    def read(self, width: int):
        if self._pos + width > len(self._bits):
            return None
        code = int(self._bits[self._pos: self._pos + width], 2)
        self._pos += width
        return code

    def exhausted(self) -> bool:
        return self._pos >= len(self._bits)


# ──────────────────────────────────────────────────────────────
# 2. LZW ENCODE
#    Règle : élargir cw APRÈS avoir ajouté nc, quand nc > 2^cw
# ──────────────────────────────────────────────────────────────

def lzw_encode_channel(channel: np.ndarray) -> bytes:
    data   = channel.ravel().tobytes()
    writer = BitWriter()

    d  = {bytes([i]): i for i in range(256)}
    nc = FIRST
    cw = MIN_BITS

    writer.write(CLEAR, cw)
    buf = b""

    for byte in data:
        nxt = buf + bytes([byte])
        if nxt in d:
            buf = nxt
        else:
            writer.write(d[buf], cw)

            if nc <= (1 << MAX_BITS) - 1:
                d[nxt] = nc
                nc += 1
                # Élargir APRÈS add, quand nc dépasse la capacité courante
                if nc > (1 << cw) and cw < MAX_BITS:
                    cw += 1
            else:
                # Dict plein → reset
                writer.write(CLEAR, cw)
                d  = {bytes([i]): i for i in range(256)}
                nc = FIRST
                cw = MIN_BITS

            buf = bytes([byte])

    if buf:
        writer.write(d[buf], cw)

    writer.write(STOP, cw)
    return writer.flush()


# ──────────────────────────────────────────────────────────────
# 3. LZW DECODE
#    Règle : élargir cw AVANT de lire, quand nc == 2^cw
#    (compense le décalage d'une itération vs l'encodeur)
# ──────────────────────────────────────────────────────────────

def lzw_decode_channel(stream: bytes, shape: tuple) -> np.ndarray:
    reader = BitReader(stream)

    d  = {i: bytes([i]) for i in range(256)}
    nc = FIRST
    cw = MIN_BITS

    out = bytearray()

    # Lire et ignorer le(s) CLEAR d'ouverture
    code = reader.read(cw)
    while code == CLEAR:
        code = reader.read(cw)

    if code is None or code == STOP:
        return np.zeros(shape, dtype=np.uint8)

    out  += d[code]
    prev  = code
    total = shape[0] * shape[1]

    while not reader.exhausted() and len(out) < total:
        # Élargir AVANT de lire si nc a atteint la limite courante
        if nc == (1 << cw) and cw < MAX_BITS:
            cw += 1

        code = reader.read(cw)
        if code is None or code == STOP:
            break

        if code == CLEAR:
            d  = {i: bytes([i]) for i in range(256)}
            nc = FIRST
            cw = MIN_BITS
            code = reader.read(cw)
            while code == CLEAR:
                code = reader.read(cw)
            if code is None or code == STOP:
                break
            out  += d[code]
            prev  = code
            continue

        # Résolution du code
        if code in d:
            entry = d[code]
        elif code == nc:
            # Cas classique LZW : code pas encore dans le dict
            entry = d[prev] + d[prev][:1]
        else:
            raise ValueError(f"Code LZW inconnu : {code}  nc={nc}  cw={cw}")

        out += entry

        if nc <= (1 << MAX_BITS) - 1:
            d[nc] = d[prev] + entry[:1]
            nc   += 1

        prev = code

    arr = np.frombuffer(bytes(out[:total]), dtype=np.uint8)
    if len(arr) < total:
        arr = np.pad(arr, (0, total - len(arr)))
    return arr.reshape(shape)


# ──────────────────────────────────────────────────────────────
# 4. COMPRESSION / DÉCOMPRESSION IMAGE
# ──────────────────────────────────────────────────────────────

def compress_image(img: np.ndarray):
    H, W, C = img.shape
    streams = {}
    total_bytes = 0
    for c in range(C):
        s = lzw_encode_channel(img[:, :, c])
        streams[c]  = s
        total_bytes += len(s)

    original_bytes = H * W * C
    bpp   = (total_bytes * 8) / (H * W)
    ratio = original_bytes / max(total_bytes, 1)
    stats = {
        "original_bytes":    original_bytes,
        "compressed_bytes":  total_bytes,
        "bpp":               bpp,
        "compression_ratio": ratio,
    }
    return streams, stats


def decompress_image(streams: dict, shape: tuple) -> np.ndarray:
    H, W, C = shape
    return np.stack(
        [lzw_decode_channel(streams[c], (H, W)) for c in range(C)],
        axis=2
    )


# ──────────────────────────────────────────────────────────────
# 5. MÉTRIQUES
# ──────────────────────────────────────────────────────────────

def compute_metrics(original, reconstructed, stats):
    psnr = psnr_func(original, reconstructed, data_range=255)
    ssim = ssim_func(original, reconstructed,
                     channel_axis=2, data_range=255)
    return {
        "psnr_db":           round(psnr, 4),
        "ssim":              round(ssim, 6),
        "bpp":               round(stats["bpp"], 4),
        "compression_ratio": round(stats["compression_ratio"], 4),
        "original_kb":       round(stats["original_bytes"] / 1024, 2),
        "compressed_kb":     round(stats["compressed_bytes"] / 1024, 2),
    }


# ──────────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────

def find_images(path, max_n):
    exts  = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    return sorted(files)[:max_n]


def load_and_crop(path, crop):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    if W > crop and H > crop:
        l = (W - crop) // 2
        t = (H - crop) // 2
        img = img.crop((l, t, l + crop, t + crop))
    return np.array(img, dtype=np.uint8)


def run_pipeline():
    base = KAGGLE_PATH if os.path.isdir(KAGGLE_PATH) else FALLBACK_PATH
    print(f"📂  Dossier source : {base}")

    files = find_images(base, MAX_IMAGES)
    if not files:
        raise FileNotFoundError(f"Aucune image trouvée dans : {base}")

    print(f"🖼️   {len(files)} images trouvées — crop {CROP_SIZE}×{CROP_SIZE}")

    records     = []
    sample_orig = sample_recon = sample_name = None

    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            img = load_and_crop(fpath, CROP_SIZE)
            t0  = time.time()
            streams, cstats = compress_image(img)
            recon           = decompress_image(streams, img.shape)
            elapsed         = time.time() - t0

            metrics = compute_metrics(img, recon, cstats)
            metrics.update({"filename": fname, "index": idx + 1,
                            "time_sec": round(elapsed, 3),
                            "width": img.shape[1], "height": img.shape[0]})
            records.append(metrics)

            if idx == 0:
                sample_orig, sample_recon, sample_name = (
                    img.copy(), recon.copy(), fname)

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1:4d}/{len(files)}]  {fname:30s}"
                      f"  PSNR={metrics['psnr_db']:7.2f} dB"
                      f"  SSIM={metrics['ssim']:.5f}"
                      f"  BPP={metrics['bpp']:.3f}"
                      f"  Ratio={metrics['compression_ratio']:.3f}x")

        except Exception as e:
            print(f"  ⚠️  {fname} — erreur : {e}")

    df = pd.DataFrame(records)[[
        "index", "filename", "width", "height",
        "psnr_db", "ssim", "bpp", "compression_ratio",
        "original_kb", "compressed_kb", "time_sec"
    ]]
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    df.to_csv(csv_path, index=False)
    print(f"\n✅  CSV exporté → {csv_path}  ({len(df)} lignes)")
    return df, sample_orig, sample_recon, sample_name


# ──────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

DARK_BG  = "#0d1117"
ACCENT1  = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
ACCENT4  = "#d2a8ff"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": "#161b22",
    "axes.edgecolor":   GRID_CLR, "axes.labelcolor": TEXT_CLR,
    "axes.titlecolor":  TEXT_CLR, "xtick.color": TEXT_CLR,
    "ytick.color":      TEXT_CLR, "grid.color": GRID_CLR,
    "text.color":       TEXT_CLR, "legend.facecolor": "#161b22",
    "legend.edgecolor": GRID_CLR, "font.family": "monospace",
})


def plot_original_vs_reconstructed(orig, recon, fname, df):
    diff = np.abs(orig.astype(int) - recon.astype(int)).astype(np.uint8)
    row  = df[df["filename"] == fname].iloc[0]

    fig = plt.figure(figsize=(18, 7), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.3,
                            left=0.04, right=0.98,
                            top=0.88, bottom=0.08)
    for ax, data, title, color in [
        (fig.add_subplot(gs[:, 0]), orig,
         "ORIGINAL", ACCENT1),
        (fig.add_subplot(gs[:, 1]), recon,
         "LZW RECONSTRUCTED", ACCENT2),
        (fig.add_subplot(gs[:, 2]), np.clip(diff * 4, 0, 255),
         "DIFF ×4 AMPLIFIED", ACCENT3),
    ]:
        ax.imshow(data)
        ax.set_title(title, fontsize=11, color=color,
                     fontweight="bold", pad=8)
        ax.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    info   = [("PSNR",       f"{row.psnr_db:.2f} dB"),
              ("SSIM",       f"{row.ssim:.5f}"),
              ("BPP",        f"{row.bpp:.3f}"),
              ("Ratio",      f"{row.compression_ratio:.3f}×"),
              ("Original",   f"{row.original_kb:.1f} KB"),
              ("Compressed", f"{row.compressed_kb:.1f} KB"),
              ("Time",       f"{row.time_sec:.3f} s")]
    colors = [ACCENT1, ACCENT2, ACCENT4, ACCENT3,
              TEXT_CLR, TEXT_CLR, TEXT_CLR]
    y = 0.95
    for (lbl, val), col in zip(info, colors):
        ax3.text(0.05, y, f"{lbl:<12}", transform=ax3.transAxes,
                 fontsize=10, color="#8b949e", va="top")
        ax3.text(0.55, y, val, transform=ax3.transAxes,
                 fontsize=10, color=col, va="top", fontweight="bold")
        y -= 0.135

    ax4 = fig.add_subplot(gs[1, 3])
    for c, (col, lbl) in enumerate(
            zip(["#f78166", "#3fb950", "#58a6ff"], ["R", "G", "B"])):
        ax4.plot(np.histogram(orig[:, :, c], bins=64,
                              range=(0, 255))[0],
                 color=col, lw=1.2, alpha=0.85, label=lbl)
    ax4.set_title("Pixel Histogram", fontsize=9, pad=5)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.set_xlim(0, 63)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7)

    fig.suptitle(f"LZW Compression  |  {fname}  |  "
                 f"{orig.shape[1]}×{orig.shape[0]} px",
                 fontsize=13, color=TEXT_CLR, y=0.97, fontweight="bold")
    out = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Comparaison → {out}")


def plot_metrics_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("LZW Compression — Metrics Dashboard",
                 fontsize=16, color=TEXT_CLR, fontweight="bold", y=0.98)
    x = df["index"].values

    def _plot(ax, y, color, title, ylabel, unit=""):
        ax.scatter(x, y, color=color, s=4, alpha=0.6, zorder=3)
        if len(y) >= 20:
            ma = pd.Series(y).rolling(20, center=True).mean()
            ax.plot(x, ma, color="white", lw=1.5, alpha=0.9,
                    label="Moving avg (20)", zorder=4)
        ax.axhline(np.mean(y), color=color, lw=1, linestyle="--",
                   alpha=0.6, label=f"Mean: {np.mean(y):.3f}{unit}")
        ax.set_title(title, fontsize=12, color=color,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Image Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0,0], df["psnr_db"].values,          ACCENT1, "PSNR", "PSNR (dB)", " dB")
    _plot(axes[0,1], df["ssim"].values,              ACCENT2, "SSIM", "SSIM Score")
    _plot(axes[1,0], df["bpp"].values,               ACCENT4, "BPP",  "BPP", " bpp")
    _plot(axes[1,1], df["compression_ratio"].values, ACCENT3, "Compression Ratio", "Ratio (×)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "metrics_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Dashboard → {out}")


def plot_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=DARK_BG)
    fig.suptitle("Metric Distributions", fontsize=14,
                 color=TEXT_CLR, fontweight="bold")
    for ax, (col, color, label) in zip(axes, [
        ("psnr_db",           ACCENT1, "PSNR (dB)"),
        ("ssim",              ACCENT2, "SSIM"),
        ("bpp",               ACCENT4, "BPP"),
        ("compression_ratio", ACCENT3, "Ratio (×)"),
    ]):
        vals = df[col].dropna().values
        ax.hist(vals, bins=30, color=color, alpha=0.75,
                edgecolor="none", density=True)
        kde = gaussian_kde(vals, bw_method=0.3)
        xs  = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), color="white", lw=2)
        ax.axvline(np.mean(vals), color=color, lw=1.5,
                   linestyle="--", label=f"μ={np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color="white", lw=1,
                   linestyle=":", label=f"med={np.median(vals):.3f}")
        ax.set_title(label, fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Density", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Distributions → {out}")


def print_summary(df):
    print("\n" + "═" * 62)
    print("  RÉSUMÉ STATISTIQUE — LZW COMPRESSION")
    print("═" * 62)
    for col, lbl, unit in [
        ("psnr_db",           "PSNR",  " dB"),
        ("ssim",              "SSIM",  ""),
        ("bpp",               "BPP",   " bpp"),
        ("compression_ratio", "Ratio", "×"),
        ("time_sec",          "Temps", " s"),
    ]:
        v = df[col]
        print(f"  {lbl:<8}  mean={v.mean():.4f}{unit}  "
              f"std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")
    print("═" * 62)
    print(f"  Images traitées : {len(df)}")
    print(f"  CSV             : {os.path.join(OUTPUT_DIR, CSV_NAME)}")
    print("═" * 62 + "\n")


# ──────────────────────────────────────────────────────────────
# ENTRÉE PRINCIPALE
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, orig, recon, fname = run_pipeline()
    print_summary(df)
    plot_original_vs_reconstructed(orig, recon, fname, df)
    plot_metrics_dashboard(df)
    plot_distributions(df)

    print("\n🎉  Pipeline terminé !")
    print(f"   Fichiers générés dans : {OUTPUT_DIR}/")
    print(f"   ├── {CSV_NAME}")
    print(f"   ├── comparison.png")
    print(f"   ├── metrics_dashboard.png")
    print(f"   └── distributions.png")