"""
╔══════════════════════════════════════════════════════════════╗
║   RLE IMAGE COMPRESSION — KAGGLE OPTIMIZED                  ║
║   ✔ Vraie compression RLE (encode/decode bitstream)         ║
║   ✔ Reconstruction image réelle                              ║
║   ✔ Table Pandas 800 images + export CSV                    ║
║   ✔ Graphes propres (PSNR, SSIM, BPP)                       ║
║   ✔ Affichage original vs reconstructed                     ║
╚══════════════════════════════════════════════════════════════╝
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
from collections import Counter
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
OUTPUT_DIR    = "./rle_output"
CSV_NAME      = "rle_metrics.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. RLE ENCODE / DECODE — VRAI BITSTREAM
# ──────────────────────────────────────────────────────────────
# Format du bitstream par canal :
#   Header  : 4 octets — uint32 BE = nombre de paires (run, value)
#   Données : chaque paire = 2 octets (uint8 run_length, uint8 value)
#             run_length ∈ [1, 255]  → max run splité si > 255
# ──────────────────────────────────────────────────────────────

def rle_encode_channel(channel: np.ndarray) -> bytes:
    """
    Encode un canal 2D uint8 → bitstream RLE bytes.
    Scanne ligne par ligne (ordre raster).
    """
    flat = channel.ravel()
    pairs = []

    i = 0
    n = len(flat)
    while i < n:
        val = flat[i]
        run = 1
        while i + run < n and flat[i + run] == val and run < 255:
            run += 1
        pairs.append((run, int(val)))
        i += run

    # Header : nombre de paires (uint32 BE)
    header = struct.pack(">I", len(pairs))
    # Corps : séquence de (run uint8, value uint8)
    body = bytes([b for run, val in pairs for b in (run, val)])
    return header + body


def rle_decode_channel(stream: bytes, shape: tuple) -> np.ndarray:
    """
    Décode un bitstream RLE → canal uint8 de forme `shape`.
    """
    num_pairs = struct.unpack(">I", stream[:4])[0]
    body = stream[4:]

    flat = np.empty(shape[0] * shape[1], dtype=np.uint8)
    pos = 0
    for k in range(num_pairs):
        run = body[k * 2]
        val = body[k * 2 + 1]
        flat[pos:pos + run] = val
        pos += run

    return flat.reshape(shape)


# ──────────────────────────────────────────────────────────────
# 2. COMPRESSION / DÉCOMPRESSION D'UNE IMAGE
# ──────────────────────────────────────────────────────────────

def compress_image(img_array: np.ndarray):
    """
    Compresse une image (H,W,C) uint8 via RLE channel-by-channel.
    Retourne (streams, stats).
    """
    H, W, C = img_array.shape
    streams = {}
    total_compressed_bytes = 0

    for c in range(C):
        stream = rle_encode_channel(img_array[:, :, c])
        streams[c] = stream
        # Header (4 B) + body (paires × 2 B)
        total_compressed_bytes += len(stream)

    original_bytes    = H * W * C
    compressed_bytes  = total_compressed_bytes
    bpp               = (compressed_bytes * 8) / (H * W)
    ratio             = original_bytes / max(compressed_bytes, 1)

    stats = {
        "original_bits":     original_bytes * 8,
        "compressed_bits":   compressed_bytes * 8,
        "original_bytes":    original_bytes,
        "compressed_bytes":  compressed_bytes,
        "bpp":               bpp,
        "compression_ratio": ratio,
    }
    return streams, stats


def decompress_image(streams: dict, shape: tuple) -> np.ndarray:
    """Reconstruit l'image depuis les bitstreams RLE."""
    H, W, C = shape
    channels = []
    for c in range(C):
        ch = rle_decode_channel(streams[c], (H, W))
        channels.append(ch)
    return np.stack(channels, axis=2)


# ──────────────────────────────────────────────────────────────
# 3. MÉTRIQUES
# ──────────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray,
                    reconstructed: np.ndarray,
                    stats: dict) -> dict:
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
# 4. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────

def find_images(path: str, max_n: int) -> list:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    return sorted(files)[:max_n]


def load_and_crop(path: str, crop: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    if W > crop and H > crop:
        left = (W - crop) // 2
        top  = (H - crop) // 2
        img  = img.crop((left, top, left + crop, top + crop))
    return np.array(img, dtype=np.uint8)


def run_pipeline():
    base_path = KAGGLE_PATH if os.path.isdir(KAGGLE_PATH) else FALLBACK_PATH
    print(f"📂  Dossier source : {base_path}")

    files = find_images(base_path, MAX_IMAGES)
    if not files:
        raise FileNotFoundError(
            f"Aucune image trouvée dans : {base_path}\n"
            "Vérifiez le chemin ou ajoutez des images dans ./DIV2K_train_HR"
        )

    print(f"🖼️   {len(files)} images trouvées — crop {CROP_SIZE}×{CROP_SIZE}")

    records      = []
    sample_orig  = None
    sample_recon = None
    sample_name  = None

    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            img = load_and_crop(fpath, CROP_SIZE)
            t0  = time.time()

            streams, cstats = compress_image(img)
            recon           = decompress_image(streams, img.shape)
            elapsed         = time.time() - t0

            metrics = compute_metrics(img, recon, cstats)
            metrics["filename"] = fname
            metrics["index"]    = idx + 1
            metrics["time_sec"] = round(elapsed, 3)
            metrics["width"]    = img.shape[1]
            metrics["height"]   = img.shape[0]
            records.append(metrics)

            if idx == 0:
                sample_orig  = img.copy()
                sample_recon = recon.copy()
                sample_name  = fname

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
# 5. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

DARK_BG  = "#0d1117"
ACCENT1  = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
ACCENT4  = "#d2a8ff"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "axes.titlecolor":  TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "grid.color":       GRID_CLR,
    "text.color":       TEXT_CLR,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": GRID_CLR,
    "font.family":      "monospace",
})


def plot_original_vs_reconstructed(orig, recon, fname, df):
    diff = np.abs(orig.astype(int) - recon.astype(int)).astype(np.uint8)
    row  = df[df["filename"] == fname].iloc[0]

    fig = plt.figure(figsize=(18, 7), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.3,
                            left=0.04, right=0.98,
                            top=0.88, bottom=0.08)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(orig)
    ax0.set_title("ORIGINAL", fontsize=11, color=ACCENT1,
                  fontweight="bold", pad=8)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(recon)
    ax1.set_title("RLE RECONSTRUCTED", fontsize=11,
                  color=ACCENT2, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(np.clip(diff * 4, 0, 255))
    ax2.set_title("DIFF ×4 AMPLIFIED", fontsize=11,
                  color=ACCENT3, fontweight="bold", pad=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    info = [
        ("PSNR",       f"{row.psnr_db:.2f} dB"),
        ("SSIM",       f"{row.ssim:.5f}"),
        ("BPP",        f"{row.bpp:.3f}"),
        ("Ratio",      f"{row.compression_ratio:.3f}×"),
        ("Original",   f"{row.original_kb:.1f} KB"),
        ("Compressed", f"{row.compressed_kb:.1f} KB"),
        ("Time",       f"{row.time_sec:.3f} s"),
    ]
    colors = [ACCENT1, ACCENT2, ACCENT4, ACCENT3,
              TEXT_CLR, TEXT_CLR, TEXT_CLR]
    y = 0.95
    for (label, val), col in zip(info, colors):
        ax3.text(0.05, y, f"{label:<12}", transform=ax3.transAxes,
                 fontsize=10, color="#8b949e", va="top")
        ax3.text(0.55, y, val, transform=ax3.transAxes,
                 fontsize=10, color=col, va="top", fontweight="bold")
        y -= 0.135

    ax4 = fig.add_subplot(gs[1, 3])
    for c, (col, lbl) in enumerate(
            zip(["#f78166", "#3fb950", "#58a6ff"], ["R", "G", "B"])):
        ax4.plot(
            np.histogram(orig[:, :, c], bins=64, range=(0, 255))[0],
            color=col, lw=1.2, alpha=0.85, label=lbl
        )
    ax4.set_title("Pixel Histogram", fontsize=9, pad=5)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.set_xlim(0, 63)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7)

    fig.suptitle(
        f"RLE Compression  |  {fname}  |  "
        f"{orig.shape[1]}×{orig.shape[0]} px",
        fontsize=13, color=TEXT_CLR, y=0.97, fontweight="bold"
    )

    out = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Comparaison sauvegardée → {out}")


def plot_metrics_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("RLE Compression — Metrics Dashboard",
                 fontsize=16, color=TEXT_CLR, fontweight="bold", y=0.98)

    x = df["index"].values

    def _scatter_trend(ax, y, color, title, ylabel, unit=""):
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

    _scatter_trend(axes[0, 0], df["psnr_db"].values,
                   ACCENT1, "PSNR", "PSNR (dB)", " dB")
    _scatter_trend(axes[0, 1], df["ssim"].values,
                   ACCENT2, "SSIM", "SSIM Score")
    _scatter_trend(axes[1, 0], df["bpp"].values,
                   ACCENT4, "BPP (Bits Per Pixel)", "BPP", " bpp")
    _scatter_trend(axes[1, 1], df["compression_ratio"].values,
                   ACCENT3, "Compression Ratio", "Ratio (×)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "metrics_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Dashboard sauvegardé → {out}")


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=DARK_BG)
    fig.suptitle("Metric Distributions", fontsize=14,
                 color=TEXT_CLR, fontweight="bold")

    metrics = [
        ("psnr_db",           ACCENT1, "PSNR (dB)"),
        ("ssim",              ACCENT2, "SSIM"),
        ("bpp",               ACCENT4, "BPP"),
        ("compression_ratio", ACCENT3, "Ratio (×)"),
    ]

    for ax, (col, color, label) in zip(axes, metrics):
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
    print(f"💾  Distributions sauvegardées → {out}")


def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 62)
    print("  RÉSUMÉ STATISTIQUE — RLE COMPRESSION")
    print("═" * 62)
    for col, lbl, unit in [
        ("psnr_db",           "PSNR",    " dB"),
        ("ssim",              "SSIM",    ""),
        ("bpp",               "BPP",     " bpp"),
        ("compression_ratio", "Ratio",   "×"),
        ("time_sec",          "Temps",   " s"),
    ]:
        v = df[col]
        print(f"  {lbl:<8}  "
              f"mean={v.mean():.4f}{unit}  "
              f"std={v.std():.4f}  "
              f"min={v.min():.4f}  "
              f"max={v.max():.4f}")
    print("═" * 62)
    print(f"  Images traitées : {len(df)}")
    print(f"  CSV             : {os.path.join(OUTPUT_DIR, CSV_NAME)}")
    print("═" * 62 + "\n")


# ──────────────────────────────────────────────────────────────
# NOTE SUR RLE VS IMAGES NATURELLES
# ──────────────────────────────────────────────────────────────
# RLE est lossless → PSNR = inf, SSIM = 1.0 (reconstruction parfaite).
# Sur images naturelles (DIV2K), les runs sont courts → ratio < 1
# (expansion). RLE excelle sur images binaires / synthétiques.
# Le script mesure quand même BPP et ratio pour comparaison académique.
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