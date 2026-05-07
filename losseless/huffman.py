"""
╔══════════════════════════════════════════════════════════════╗
║   HUFFMAN IMAGE COMPRESSION — KAGGLE OPTIMIZED              ║
║   ✔ Vraie compression bitstream Huffman (encode/decode)     ║
║   ✔ Reconstruction image réelle                              ║
║   ✔ Table Pandas 800 images + export CSV                    ║
║   ✔ Graphes propres (PSNR, SSIM, BPP)                       ║
║   ✔ Affichage original vs reconstructed                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import glob
import heapq
import struct
import io
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import math

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
KAGGLE_PATH  = "/kaggle/input/datasets/mariyyyaaella/div2k/DIV2K_train_HR"
FALLBACK_PATH = "./DIV2K_train_HR"
MAX_IMAGES    = 800
CROP_SIZE     = 256          # crop pour accélérer sur Kaggle
OUTPUT_DIR    = "./huffman_output"
CSV_NAME      = "huffman_metrics.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. ARBRE HUFFMAN — VRAI BITSTREAM
# ──────────────────────────────────────────────────────────────

class HuffmanNode:
    __slots__ = ("symbol", "freq", "left", "right")
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freqs: dict) -> HuffmanNode:
    """Construit l'arbre Huffman depuis un dict {symbole: fréquence}."""
    heap = [HuffmanNode(sym, f) for sym, f in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return HuffmanNode(None, node.freq, left=node)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))
    return heap[0]


def build_codebook(root: HuffmanNode) -> dict:
    """Génère le codebook {symbole: bitstring}."""
    codebook = {}
    def traverse(node, bits=""):
        if node is None:
            return
        if node.symbol is not None:
            codebook[node.symbol] = bits or "0"
        else:
            traverse(node.left,  bits + "0")
            traverse(node.right, bits + "1")
    traverse(root)
    return codebook


def encode_bitstream(data: np.ndarray, codebook: dict) -> bytes:
    """
    Encode un tableau 1-D uint8 → vrai bitstream bytes.
    Format:
      - 4 octets : longueur totale en bits (uint32 big-endian)
      - N octets : bits packés (MSB first, padding 0 à la fin)
    """
    bitstring = "".join(codebook[b] for b in data.ravel())
    total_bits = len(bitstring)
    # Padding à multiple de 8
    pad = (8 - total_bits % 8) % 8
    bitstring += "0" * pad
    # Pack en bytes
    packed = bytearray()
    for i in range(0, len(bitstring), 8):
        packed.append(int(bitstring[i:i+8], 2))
    # Préfixe 4 octets = nb de bits utiles
    header = struct.pack(">I", total_bits)
    return header + bytes(packed)


def decode_bitstream(stream: bytes, root: HuffmanNode,
                     original_len: int, shape: tuple) -> np.ndarray:
    """
    Décode un bitstream → tableau uint8 de forme `shape`.
    """
    total_bits = struct.unpack(">I", stream[:4])[0]
    packed = stream[4:]
    # Dépacker les bits
    bits = "".join(f"{byte:08b}" for byte in packed)[:total_bits]

    symbols = []
    node = root
    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            symbols.append(node.symbol)
            node = root
            if len(symbols) == original_len:
                break

    return np.array(symbols, dtype=np.uint8).reshape(shape)


# ──────────────────────────────────────────────────────────────
# 2. COMPRESSION / DÉCOMPRESSION D'UNE IMAGE
# ──────────────────────────────────────────────────────────────

def compress_image(img_array: np.ndarray):
    """
    Compresse une image (H,W,C) uint8 via Huffman channel-by-channel.
    Retourne (streams_dict, codebooks_dict, trees_dict, stats_dict).
    """
    H, W, C = img_array.shape
    streams   = {}
    codebooks = {}
    trees     = {}
    total_compressed_bits = 0

    for c in range(C):
        channel = img_array[:, :, c]
        freqs   = Counter(channel.ravel().tolist())
        root    = build_huffman_tree(freqs)
        cb      = build_codebook(root)
        stream  = encode_bitstream(channel, cb)

        streams[c]   = stream
        codebooks[c] = cb
        trees[c]     = root

        # Bits du bitstream utiles (extrait du header)
        total_compressed_bits += struct.unpack(">I", stream[:4])[0]

    original_bits = H * W * C * 8
    bpp           = total_compressed_bits / (H * W)
    ratio         = original_bits / max(total_compressed_bits, 1)

    stats = {
        "original_bits":    original_bits,
        "compressed_bits":  total_compressed_bits,
        "bpp":              bpp,
        "compression_ratio": ratio,
    }
    return streams, codebooks, trees, stats


def decompress_image(streams: dict, trees: dict,
                     shape: tuple) -> np.ndarray:
    """Reconstruit l'image depuis les bitstreams."""
    H, W, C = shape
    channels = []
    for c in range(C):
        ch = decode_bitstream(streams[c], trees[c], H * W, (H, W))
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
        "psnr_db":          round(psnr, 4),
        "ssim":             round(ssim, 6),
        "bpp":              round(stats["bpp"], 4),
        "compression_ratio": round(stats["compression_ratio"], 4),
        "original_kb":      round(stats["original_bits"] / 8 / 1024, 2),
        "compressed_kb":    round(stats["compressed_bits"] / 8 / 1024, 2),
    }


# ──────────────────────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────

def find_images(path: str, max_n: int) -> list:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    files = sorted(files)[:max_n]
    return files


def load_and_crop(path: str, crop: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    if W > crop and H > crop:
        # Centre crop
        left = (W - crop) // 2
        top  = (H - crop) // 2
        img  = img.crop((left, top, left + crop, top + crop))
    return np.array(img, dtype=np.uint8)


def run_pipeline():
    # Localise les images
    base_path = KAGGLE_PATH if os.path.isdir(KAGGLE_PATH) else FALLBACK_PATH
    print(f"📂  Dossier source : {base_path}")

    files = find_images(base_path, MAX_IMAGES)
    if not files:
        raise FileNotFoundError(
            f"Aucune image trouvée dans : {base_path}\n"
            f"Vérifiez le chemin ou ajoutez des images dans ./DIV2K_train_HR"
        )

    print(f"🖼️   {len(files)} images trouvées — crop {CROP_SIZE}×{CROP_SIZE}")

    records        = []
    sample_orig    = None
    sample_recon   = None
    sample_name    = None

    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            img = load_and_crop(fpath, CROP_SIZE)
            t0  = time.time()
            streams, cbs, trees, cstats = compress_image(img)
            recon = decompress_image(streams, trees, img.shape)
            elapsed = time.time() - t0

            metrics = compute_metrics(img, recon, cstats)
            metrics["filename"]    = fname
            metrics["index"]       = idx + 1
            metrics["time_sec"]    = round(elapsed, 3)
            metrics["width"]       = img.shape[1]
            metrics["height"]      = img.shape[0]
            records.append(metrics)

            # Garde la première image pour le plot
            if idx == 0:
                sample_orig  = img.copy()
                sample_recon = recon.copy()
                sample_name  = fname

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1:4d}/{len(files)}]  {fname:30s}"
                      f"  PSNR={metrics['psnr_db']:6.2f} dB"
                      f"  SSIM={metrics['ssim']:.4f}"
                      f"  BPP={metrics['bpp']:.3f}"
                      f"  Ratio={metrics['compression_ratio']:.2f}x")

        except Exception as e:
            print(f"  ⚠️  {fname} — erreur : {e}")

    df = pd.DataFrame(records)
    cols = ["index", "filename", "width", "height",
            "psnr_db", "ssim", "bpp", "compression_ratio",
            "original_kb", "compressed_kb", "time_sec"]
    df = df[cols]

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    df.to_csv(csv_path, index=False)
    print(f"\n✅  CSV exporté → {csv_path}  ({len(df)} lignes)")

    return df, sample_orig, sample_recon, sample_name


# ──────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
ACCENT1   = "#58a6ff"
ACCENT2   = "#3fb950"
ACCENT3   = "#f78166"
ACCENT4   = "#d2a8ff"
TEXT_CLR  = "#e6edf3"
GRID_CLR  = "#21262d"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "grid.color":        GRID_CLR,
    "text.color":        TEXT_CLR,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  GRID_CLR,
    "font.family":       "monospace",
})


def plot_original_vs_reconstructed(orig, recon, fname, df):
    """Affiche original / reconstruit + diff + histogramme."""
    diff = np.abs(orig.astype(int) - recon.astype(int)).astype(np.uint8)
    row  = df[df["filename"] == fname].iloc[0]

    fig = plt.figure(figsize=(18, 7), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.3,
                            left=0.04, right=0.98,
                            top=0.88, bottom=0.08)

    # Original
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(orig)
    ax0.set_title("ORIGINAL", fontsize=11, color=ACCENT1,
                  fontweight="bold", pad=8)
    ax0.axis("off")

    # Reconstructed
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(recon)
    ax1.set_title("HUFFMAN RECONSTRUCTED", fontsize=11,
                  color=ACCENT2, fontweight="bold", pad=8)
    ax1.axis("off")

    # Diff map (amplifié ×4)
    ax2 = fig.add_subplot(gs[:, 2])
    diff_amp = np.clip(diff * 4, 0, 255)
    ax2.imshow(diff_amp)
    ax2.set_title("DIFF ×4 AMPLIFIED", fontsize=11,
                  color=ACCENT3, fontweight="bold", pad=8)
    ax2.axis("off")

    # Métriques texte
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    info = [
        ("PSNR",       f"{row.psnr_db:.2f} dB"),
        ("SSIM",       f"{row.ssim:.5f}"),
        ("BPP",        f"{row.bpp:.3f}"),
        ("Ratio",      f"{row.compression_ratio:.2f}×"),
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

    # Histogramme des canaux
    ax4 = fig.add_subplot(gs[1, 3])
    chan_colors = ["#f78166", "#3fb950", "#58a6ff"]
    labels_rgb  = ["R", "G", "B"]
    for c, (col, lbl) in enumerate(zip(chan_colors, labels_rgb)):
        ax4.plot(np.histogram(orig[:, :, c], bins=64, range=(0, 255))[0],
                 color=col, lw=1.2, alpha=0.85, label=lbl)
    ax4.set_title("Pixel Histogram", fontsize=9, pad=5)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.set_xlim(0, 63)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7)

    fig.suptitle(
        f"Huffman Compression  |  {fname}  |  "
        f"{orig.shape[1]}×{orig.shape[0]} px",
        fontsize=13, color=TEXT_CLR, y=0.97, fontweight="bold"
    )

    out = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.show()
    print(f"💾  Comparaison sauvegardée → {out}")


def plot_metrics_dashboard(df: pd.DataFrame):
    """4 graphes : PSNR, SSIM, BPP, Ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("Huffman Compression — Metrics Dashboard",
                 fontsize=16, color=TEXT_CLR, fontweight="bold", y=0.98)

    x = df["index"].values

    def _scatter_with_trend(ax, y, color, title, ylabel, unit=""):
        ax.scatter(x, y, color=color, s=4, alpha=0.6, zorder=3)
        # Moyenne glissante 20
        if len(y) >= 20:
            ma = pd.Series(y).rolling(20, center=True).mean()
            ax.plot(x, ma, color="white", lw=1.5, alpha=0.9,
                    label="Moving avg (20)", zorder=4)
        ax.axhline(np.mean(y), color=color, lw=1,
                   linestyle="--", alpha=0.6,
                   label=f"Mean: {np.mean(y):.3f}{unit}")
        ax.set_title(title, fontsize=12, color=color,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Image Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _scatter_with_trend(axes[0, 0], df["psnr_db"].values,
                        ACCENT1, "PSNR", "PSNR (dB)", " dB")
    _scatter_with_trend(axes[0, 1], df["ssim"].values,
                        ACCENT2, "SSIM", "SSIM Score")
    _scatter_with_trend(axes[1, 0], df["bpp"].values,
                        ACCENT4, "BPP (Bits Per Pixel)", "BPP", " bpp")
    _scatter_with_trend(axes[1, 1], df["compression_ratio"].values,
                        ACCENT3, "Compression Ratio", "Ratio (×)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "metrics_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  Dashboard sauvegardé → {out}")


def plot_distributions(df: pd.DataFrame):
    """Distributions (histogrammes + KDE) des 4 métriques."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=DARK_BG)
    fig.suptitle("Metric Distributions", fontsize=14,
                 color=TEXT_CLR, fontweight="bold")

    metrics = [
        ("psnr_db",          ACCENT1, "PSNR (dB)"),
        ("ssim",             ACCENT2, "SSIM"),
        ("bpp",              ACCENT4, "BPP"),
        ("compression_ratio", ACCENT3, "Ratio (×)"),
    ]

    for ax, (col, color, label) in zip(axes, metrics):
        vals = df[col].dropna().values
        ax.hist(vals, bins=30, color=color, alpha=0.75,
                edgecolor="none", density=True)
        # KDE simple (gaussien)
        from scipy.stats import gaussian_kde
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
    print("\n" + "═"*60)
    print("  RÉSUMÉ STATISTIQUE — HUFFMAN COMPRESSION")
    print("═"*60)
    for col, lbl, unit in [
        ("psnr_db",           "PSNR",    "dB"),
        ("ssim",              "SSIM",    ""),
        ("bpp",               "BPP",     "bpp"),
        ("compression_ratio", "Ratio",   "×"),
        ("time_sec",          "Temps",   "s"),
    ]:
        v = df[col]
        print(f"  {lbl:<8}  "
              f"mean={v.mean():.4f}{unit}  "
              f"std={v.std():.4f}  "
              f"min={v.min():.4f}  "
              f"max={v.max():.4f}")
    print("═"*60)
    print(f"  Images traitées : {len(df)}")
    print(f"  CSV             : {os.path.join(OUTPUT_DIR, CSV_NAME)}")
    print("═"*60 + "\n")


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