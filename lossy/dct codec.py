"""
╔══════════════════════════════════════════════════════════════════╗
║   JPEG DCT IMAGE COMPRESSION — v2                               ║
║   3-WAY COMPARISON:                                             ║
║     1. Your DCT Codec                                           ║
║     2. Baseline OpenCV JPEG (matching BPP)                      ║
║     3. Optimized JPEG via PIL optimize=True (matching BPP)      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import glob
import heapq
import struct
import time
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from PIL import Image
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from scipy.stats import gaussian_kde

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("⚠️  cv2 not found — baseline JPEG comparison disabled. "
          "Install with: pip install opencv-python-headless")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
KAGGLE_PATH    = "/kaggle/input/datasets/mariyyyaaella/div2k/DIV2K_train_HR"
FALLBACK_PATH  = "./DIV2K_train_HR"
MAX_IMAGES     = 800
CROP_SIZE      = 256
OUTPUT_DIR     = "./jpeg_dct_output"
CSV_NAME       = "jpeg_dct_metrics.csv"
QUALITY_LEVELS = [10, 25, 50, 75, 90]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 0. CONFIDENCE INTERVAL UTILITIES
# ──────────────────────────────────────────────────────────────
#
# We use the BCa (bias-corrected and accelerated) bootstrap, which is
# second-order accurate and handles skewed distributions — important here
# because PSNR and SSIM are both bounded metrics with non-Gaussian tails.
#
# Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap",
#            Ch. 14.
#
# CI_ALPHA  : significance level  (0.05  →  95 % CI)
# CI_N_BOOT : number of bootstrap resamples (2000 is standard for papers)
# ──────────────────────────────────────────────────────────────

CI_ALPHA  = 0.05        # 95 % confidence intervals
CI_N_BOOT = 2000        # bootstrap resamples
CI_SEED   = 42          # reproducibility

_rng = np.random.default_rng(CI_SEED)


def bootstrap_bca(data: np.ndarray,
                  statistic=np.mean,
                  n_boot: int = CI_N_BOOT,
                  alpha: float = CI_ALPHA) -> tuple[float, float, float]:
    """
    BCa bootstrap confidence interval for a scalar statistic.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data, dtype=np.float64)
    n    = len(data)
    if n < 2:
        t = statistic(data)
        return float(t), float(t), float(t)

    theta_hat = statistic(data)

    # ── bootstrap distribution ────────────────────────────────
    boot_idx  = _rng.integers(0, n, size=(n_boot, n))
    boot_dist = np.array([statistic(data[idx]) for idx in boot_idx])

    # ── bias-correction factor z0 ─────────────────────────────
    prop_less = np.mean(boot_dist < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    from scipy.stats import norm as _norm
    z0 = _norm.ppf(prop_less)

    # ── acceleration factor a (jackknife) ─────────────────────
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    num   = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a     = num / denom if denom != 0 else 0.0

    # ── adjusted quantile levels ──────────────────────────────
    z_lo = _norm.ppf(alpha / 2)
    z_hi = _norm.ppf(1 - alpha / 2)

    def _adj(z):
        return _norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo_q = np.clip(_adj(z_lo), 0, 1)
    hi_q = np.clip(_adj(z_hi), 0, 1)

    ci_lo = float(np.quantile(boot_dist, lo_q))
    ci_hi = float(np.quantile(boot_dist, hi_q))
    return float(theta_hat), ci_lo, ci_hi


def compute_ci_table(df: pd.DataFrame,
                     metrics: list[str] | None = None,
                     alpha: float = CI_ALPHA) -> pd.DataFrame:
    """
    For each quality level × metric, compute mean + BCa 95 % CI.

    Returns a DataFrame with columns:
        quality, metric, mean, ci_lo, ci_hi, ci_half_width, n
    """
    if metrics is None:
        metrics = ["psnr_db", "ssim", "bpp", "compression_ratio"]

    rows = []
    for q in sorted(df["quality"].unique()):
        sub = df[df["quality"] == q]
        for col in metrics:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().values
            mean, lo, hi = bootstrap_bca(vals, alpha=alpha)
            rows.append({
                "quality":       q,
                "metric":        col,
                "mean":          round(mean, 6),
                "ci_lo":         round(lo,   6),
                "ci_hi":         round(hi,   6),
                "ci_half_width": round((hi - lo) / 2, 6),
                "n":             len(vals),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 1. QUANTISATION TABLES
# ──────────────────────────────────────────────────────────────

QUANT_LUMA = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

QUANT_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

ZIGZAG_IDX = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
])
IZIGZAG_IDX = np.argsort(ZIGZAG_IDX)


def scale_quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    if quality <= 0:  quality = 1
    if quality > 100: quality = 100
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = np.floor((base * scale + 50) / 100).astype(np.float32)
    return np.clip(table, 1, 255)


# ──────────────────────────────────────────────────────────────
# 2. COLOUR CONVERSION
# ──────────────────────────────────────────────────────────────

def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    Y  =  0.299    * r + 0.587    * g + 0.114    * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5      * b
    Cr =  0.5      * r - 0.418688 * g - 0.081312 * b
    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]
    r = Y                  + 1.402    * Cr
    g = Y - 0.344136 * Cb  - 0.714136 * Cr
    b = Y + 1.772    * Cb
    return np.clip(np.round(np.stack([r, g, b], axis=2)), 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# 3. 4:2:0 CHROMA SUBSAMPLING
# ──────────────────────────────────────────────────────────────

def downsample_420(channel: np.ndarray) -> np.ndarray:
    H, W = channel.shape
    H2, W2 = H // 2, W // 2
    return (channel[:H2*2:2, :W2*2:2] + channel[1:H2*2:2, :W2*2:2] +
            channel[:H2*2:2, 1:W2*2:2] + channel[1:H2*2:2, 1:W2*2:2]) / 4.0


def upsample_420(channel: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    up = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return up[:target_H, :target_W]


# ──────────────────────────────────────────────────────────────
# 4. DCT + QUANTISATION
# ──────────────────────────────────────────────────────────────

def pad_to_multiple(channel: np.ndarray, block: int = 8) -> np.ndarray:
    H, W = channel.shape
    pH = (block - H % block) % block
    pW = (block - W % block) % block
    return np.pad(channel, ((0, pH), (0, pW)), mode="edge")


def encode_channel(channel: np.ndarray,
                   quant_table: np.ndarray,
                   dc_offset: float = 128.0) -> tuple:
    H, W = channel.shape
    ch_pad = pad_to_multiple(channel - dc_offset)
    Hp, Wp = ch_pad.shape
    coeffs = []
    for bh in range(Hp // 8):
        for bw in range(Wp // 8):
            block = ch_pad[bh*8:(bh+1)*8, bw*8:(bw+1)*8]
            dct   = dctn(block, norm="ortho")
            quant = np.round(dct / quant_table).astype(np.int16)
            zz    = quant.ravel()[ZIGZAG_IDX]
            coeffs.append(zz)
    return coeffs, (Hp, Wp), (H, W), dc_offset


def decode_channel(coeffs: list,
                   quant_table: np.ndarray,
                   shape_padded: tuple,
                   shape_orig: tuple,
                   dc_offset: float = 128.0) -> np.ndarray:
    Hp, Wp = shape_padded
    H,  W  = shape_orig
    blocks_W = Wp // 8
    ch_pad = np.zeros((Hp, Wp), dtype=np.float32)
    for idx, zz in enumerate(coeffs):
        bh = idx // blocks_W
        bw = idx  % blocks_W
        quant = zz[IZIGZAG_IDX].reshape(8, 8).astype(np.float32)
        dct   = quant * quant_table
        block = idctn(dct, norm="ortho")
        ch_pad[bh*8:(bh+1)*8, bw*8:(bw+1)*8] = block
    return ch_pad[:H, :W] + dc_offset


# ──────────────────────────────────────────────────────────────
# 5. HUFFMAN
# ──────────────────────────────────────────────────────────────

class HuffNode:
    __slots__ = ("sym", "freq", "l", "r")
    def __init__(self, sym, freq, l=None, r=None):
        self.sym, self.freq, self.l, self.r = sym, freq, l, r
    def __lt__(self, o): return self.freq < o.freq


def build_huffman(freqs):
    heap = [HuffNode(s, f) for s, f in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        n = heapq.heappop(heap)
        return HuffNode(None, n.freq, l=n)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffNode(None, a.freq + b.freq, a, b))
    return heap[0]


def build_codebook(root):
    cb = {}
    def _walk(node, bits=""):
        if node is None: return
        if node.sym is not None: cb[node.sym] = bits or "0"
        else: _walk(node.l, bits+"0"); _walk(node.r, bits+"1")
    _walk(root)
    return cb


def pack_bits(bitstring: str) -> bytes:
    pad = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad
    total_bits = len(bitstring) - pad
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i+8], 2))
    return struct.pack(">I", total_bits) + bytes(out)


def unpack_bits(data: bytes) -> str:
    total = struct.unpack(">I", data[:4])[0]
    bits  = "".join(f"{b:08b}" for b in data[4:])
    return bits[:total]


def huffman_encode_symbols(symbols: list) -> tuple:
    freqs = Counter(symbols)
    root  = build_huffman(freqs)
    cb    = build_codebook(root)
    bs    = "".join(cb[s] for s in symbols)
    return pack_bits(bs), root


def huffman_decode_symbols(data: bytes, root, n: int) -> list:
    bits = unpack_bits(data)
    syms, node = [], root
    for bit in bits:
        node = node.l if bit == "0" else node.r
        if node.sym is not None:
            syms.append(node.sym)
            node = root
            if len(syms) == n: break
    return syms


# ──────────────────────────────────────────────────────────────
# 6. RLE — FIXED
# ──────────────────────────────────────────────────────────────

def coeffs_to_dc_ac(coeffs: list) -> tuple:
    dc_stream = []
    ac_stream = []
    prev_dc = 0
    for zz in coeffs:
        dc = int(zz[0])
        dc_stream.append(dc - prev_dc)
        prev_dc = dc

        zeros = 0
        for v in zz[1:]:
            v = int(v)
            if v == 0:
                zeros += 1
            else:
                while zeros >= 16:
                    ac_stream.append((15, 0))   # ZRL
                    zeros -= 16
                ac_stream.append((zeros, v))
                zeros = 0
        ac_stream.append((0, 0))   # EOB

    return dc_stream, ac_stream


def dc_ac_to_coeffs(dc_stream, ac_stream, n_blocks: int) -> list:
    coeffs  = []
    prev_dc = 0
    ac_iter = iter(ac_stream)

    for _ in range(n_blocks):
        zz = np.zeros(64, dtype=np.int16)
        diff = dc_stream[_]
        dc   = prev_dc + diff
        zz[0] = dc
        prev_dc = dc

        pos = 1
        while True:
            run, val = next(ac_iter)
            if run == 0 and val == 0:   # EOB
                break
            pos += run
            if val != 0 and pos < 64:
                zz[pos] = val
                pos += 1
        coeffs.append(zz)

    return coeffs


# ──────────────────────────────────────────────────────────────
# 7. COMPRESS / DECOMPRESS
# ──────────────────────────────────────────────────────────────

def compress_image(img: np.ndarray, quality: int) -> tuple:
    H, W = img.shape[:2]
    ycbcr = rgb_to_ycbcr(img)

    Y  = ycbcr[:, :, 0]
    Cb = downsample_420(ycbcr[:, :, 1])
    Cr = downsample_420(ycbcr[:, :, 2])

    qt_luma   = scale_quant_table(QUANT_LUMA,   quality)
    qt_chroma = scale_quant_table(QUANT_CHROMA, quality)

    total_bits = 0
    streams    = {}
    meta       = {}

    for name, ch, qt, offset in [
        ("Y",  Y,  qt_luma,   128.0),
        ("Cb", Cb, qt_chroma,   0.0),
        ("Cr", Cr, qt_chroma,   0.0),
    ]:
        coeffs, shape_p, shape_o, dc_off = encode_channel(ch, qt, offset)
        dc_s, ac_s = coeffs_to_dc_ac(coeffs)

        dc_bytes, dc_root = huffman_encode_symbols(dc_s)
        ac_syms           = [f"{r},{v}" for r, v in ac_s]
        ac_bytes, ac_root = huffman_encode_symbols(ac_syms)

        n_bits = (struct.unpack(">I", dc_bytes[:4])[0] +
                  struct.unpack(">I", ac_bytes[:4])[0])
        total_bits += n_bits

        streams[name] = (dc_bytes, dc_root, ac_bytes, ac_root)
        meta[name]    = {
            "shape_p":  shape_p,
            "shape_o":  shape_o,
            "dc_off":   dc_off,
            "n_blocks": len(coeffs),
            "n_dc":     len(dc_s),
            "n_ac":     len(ac_syms),
            "qt":       qt,
        }

    bpp   = total_bits / (H * W)
    ratio = (H * W * 3 * 8) / max(total_bits, 1)

    stats = {
        "original_bits":     H * W * 3 * 8,
        "compressed_bits":   total_bits,
        "bpp":               bpp,
        "compression_ratio": ratio,
    }
    return streams, meta, stats


def decompress_image(streams: dict, meta: dict, orig_shape: tuple) -> np.ndarray:
    H, W = orig_shape[:2]
    channels = {}

    for name in ("Y", "Cb", "Cr"):
        dc_bytes, dc_root, ac_bytes, ac_root = streams[name]
        m = meta[name]

        dc_s    = huffman_decode_symbols(dc_bytes, dc_root, m["n_dc"])
        ac_flat = huffman_decode_symbols(ac_bytes, ac_root, m["n_ac"])
        ac_s    = [tuple(int(x) for x in s.split(",")) for s in ac_flat]

        coeffs = dc_ac_to_coeffs(dc_s, ac_s, m["n_blocks"])
        ch     = decode_channel(coeffs, m["qt"],
                                m["shape_p"], m["shape_o"], m["dc_off"])
        channels[name] = ch

    Y_rec  = channels["Y"]
    Cb_rec = upsample_420(channels["Cb"], H, W)
    Cr_rec = upsample_420(channels["Cr"], H, W)

    return ycbcr_to_rgb(np.stack([Y_rec, Cb_rec, Cr_rec], axis=2))


# ──────────────────────────────────────────────────────────────
# 8. METRICS
# ──────────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray,
                    reconstructed: np.ndarray,
                    stats: dict) -> dict:
    psnr = psnr_func(original, reconstructed, data_range=255)
    ssim = ssim_func(original, reconstructed, channel_axis=2, data_range=255)
    return {
        "psnr_db":           round(psnr, 4),
        "ssim":              round(ssim, 6),
        "bpp":               round(stats["bpp"], 4),
        "compression_ratio": round(stats["compression_ratio"], 4),
        "original_kb":       round(stats["original_bits"] / 8 / 1024, 2),
        "compressed_kb":     round(stats["compressed_bits"] / 8 / 1024, 2),
    }


# ──────────────────────────────────────────────────────────────
# 9. JPEG HELPERS (BASELINE + OPTIMIZE)
# ──────────────────────────────────────────────────────────────

def jpeg_encode_decode_cv2(img: np.ndarray, quality: int) -> tuple:
    """
    Baseline JPEG round-trip via OpenCV (standard Huffman tables).
    Returns (reconstructed_uint8, bpp).
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 not available")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed at quality={quality}")
    n_bits = len(buf.tobytes()) * 8
    bpp    = n_bits / (img.shape[0] * img.shape[1])
    dec    = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB), bpp


def jpeg_encode_decode_pil_optimize(img: np.ndarray, quality: int) -> tuple:
    """
    Optimized JPEG round-trip via PIL with optimize=True.
    PIL's optimize flag triggers a second Huffman pass to find shorter codes
    (equivalent to libjpeg's optimized Huffman coding), producing a smaller
    file than the standard baseline tables at the same quality level.
    Returns (reconstructed_uint8, bpp).
    """
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True,
                 subsampling=2)   # subsampling=2 → 4:2:0, same as baseline
    n_bits = buf.tell() * 8
    bpp    = n_bits / (img.shape[0] * img.shape[1])
    buf.seek(0)
    recon = np.array(Image.open(buf).convert("RGB"))
    return recon, bpp


def _find_quality_for_bpp(encode_fn, img: np.ndarray,
                           target_bpp: float) -> tuple:
    """
    Binary-search the JPEG quality (1–95) whose BPP is closest to target_bpp.
    Works with any encode_fn(img, quality) -> (recon, bpp).
    Returns (quality, recon, actual_bpp).
    """
    lo, hi = 1, 95
    best_q, best_recon, best_bpp = 1, None, None

    for _ in range(12):
        mid = (lo + hi) // 2
        recon, bpp = encode_fn(img, mid)
        if best_recon is None or abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
            best_q, best_recon, best_bpp = mid, recon, bpp
        if bpp < target_bpp:
            lo = mid + 1
        else:
            hi = mid - 1

    # Fine-tune ±3
    for q in range(max(1, best_q - 3), min(95, best_q + 4)):
        recon, bpp = encode_fn(img, q)
        if abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
            best_q, best_recon, best_bpp = q, recon, bpp

    return best_q, best_recon, best_bpp


def run_jpeg_comparisons(img: np.ndarray, target_bpp: float) -> dict:
    """
    Run both JPEG variants (baseline + optimize) at the BPP closest to
    target_bpp.  Returns a flat dict of metrics ready to merge into a record.
    """
    result = {
        # ── baseline (cv2) ──
        "jpeg_q_match":   -1, "jpeg_bpp":  0.0,
        "jpeg_psnr":       0.0, "jpeg_ssim": 0.0,
        # ── optimized (PIL optimize=True) ──
        "jpeg_opt_q":     -1, "jpeg_opt_bpp": 0.0,
        "jpeg_opt_psnr":   0.0, "jpeg_opt_ssim": 0.0,
    }

    if _CV2_AVAILABLE:
        try:
            q, recon, bpp = _find_quality_for_bpp(jpeg_encode_decode_cv2,
                                                   img, target_bpp)
            result.update({
                "jpeg_q_match": q,
                "jpeg_bpp":     round(bpp, 4),
                "jpeg_psnr":    round(psnr_func(img, recon, data_range=255), 4),
                "jpeg_ssim":    round(ssim_func(img, recon, channel_axis=2,
                                                data_range=255), 6),
            })
        except Exception as e:
            print(f"    ⚠️  Baseline JPEG failed: {e}")

    try:
        q_opt, recon_opt, bpp_opt = _find_quality_for_bpp(
            jpeg_encode_decode_pil_optimize, img, target_bpp)
        result.update({
            "jpeg_opt_q":    q_opt,
            "jpeg_opt_bpp":  round(bpp_opt, 4),
            "jpeg_opt_psnr": round(psnr_func(img, recon_opt, data_range=255), 4),
            "jpeg_opt_ssim": round(ssim_func(img, recon_opt, channel_axis=2,
                                             data_range=255), 6),
        })
    except Exception as e:
        print(f"    ⚠️  Optimized JPEG failed: {e}")

    return result


# ──────────────────────────────────────────────────────────────
# 10. IMAGE LOADING
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


# ──────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline():
    base_path = KAGGLE_PATH if os.path.isdir(KAGGLE_PATH) else FALLBACK_PATH
    print(f"📂  Source: {base_path}")

    files = find_images(base_path, MAX_IMAGES)
    if not files:
        raise FileNotFoundError(f"No images found in {base_path}")

    print(f"🖼️   {len(files)} images — crop {CROP_SIZE}×{CROP_SIZE} "
          f"— qualities {QUALITY_LEVELS}")
    print("  Comparing: Your DCT Codec  |  Baseline JPEG (cv2)  |  "
          "Optimized JPEG (PIL optimize=True)")

    records     = []
    sample_data = {}
    sample_name = None

    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            img = load_and_crop(fpath, CROP_SIZE)

            for q in QUALITY_LEVELS:
                t0 = time.time()

                # ── YOUR METHOD ──────────────────────────────────────
                streams, meta, cstats = compress_image(img, q)
                recon   = decompress_image(streams, meta, img.shape)
                metrics = compute_metrics(img, recon, cstats)

                # ── BOTH JPEG VARIANTS (matched BPP) ─────────────────
                jpeg_metrics = run_jpeg_comparisons(img, metrics["bpp"])

                elapsed = time.time() - t0

                record = {
                    **metrics,
                    "filename": fname,
                    "index":    idx + 1,
                    "quality":  q,
                    "time_sec": round(elapsed, 3),
                    "width":    img.shape[1],
                    "height":   img.shape[0],
                    **jpeg_metrics,
                }
                records.append(record)

                if idx == 0:
                    sample_data[q] = (img.copy(), recon.copy())
                    sample_name    = fname

            if (idx + 1) % 25 == 0 or idx == 0:
                row = records[-1]
                print(f"  [{idx+1:4d}/{len(files)}]  {fname:25s}"
                      f"  Q={row['quality']:3d}"
                      f"  PSNR={row['psnr_db']:6.2f}  "
                      f"  BPP={row['bpp']:.3f}"
                      f"  JPEG_Q={row['jpeg_q_match']:3d}"
                      f"  OptJPEG_Q={row['jpeg_opt_q']:3d}")

        except Exception as e:
            print(f"  ⚠️  {fname} — error: {e}")

    # ── Build DataFrame ───────────────────────────────────────
    df = pd.DataFrame(records)[[
        "index", "filename", "quality", "width", "height",
        # Your codec
        "psnr_db", "ssim", "bpp", "compression_ratio",
        # Baseline JPEG
        "jpeg_q_match", "jpeg_bpp", "jpeg_psnr", "jpeg_ssim",
        # Optimized JPEG
        "jpeg_opt_q", "jpeg_opt_bpp", "jpeg_opt_psnr", "jpeg_opt_ssim",
        # Size / timing
        "original_kb", "compressed_kb", "time_sec"
    ]]

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    df.to_csv(csv_path, index=False)
    print(f"\n✅  CSV → {csv_path}  ({len(df)} rows)")
    return df, sample_data, sample_name


# ──────────────────────────────────────────────────────────────
# 12. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

DARK_BG  = "#0d1117"
ACCENT1  = "#58a6ff"   # Your codec
ACCENT2  = "#3fb950"   # Baseline JPEG
ACCENT3  = "#ffa657"   # Optimized JPEG
ACCENT4  = "#d2a8ff"
ACCENT5  = "#f78166"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"
Q_COLORS = {10: "#f78166", 25: "#ffa657", 50: "#d2a8ff",
            75: "#3fb950", 90: "#58a6ff"}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor":  "#161b22",
    "axes.edgecolor":   GRID_CLR, "axes.labelcolor": TEXT_CLR,
    "axes.titlecolor":  TEXT_CLR, "xtick.color":     TEXT_CLR,
    "ytick.color":      TEXT_CLR, "grid.color":      GRID_CLR,
    "text.color":       TEXT_CLR, "legend.facecolor":"#161b22",
    "legend.edgecolor": GRID_CLR, "font.family":     "monospace",
})


def plot_metrics_vs_quality(df: pd.DataFrame):
    """
    Line plots of mean ± BCa 95 % CI for the four core metrics of
    your DCT codec, across quality levels.
    The shaded band is the true confidence interval on the mean
    (not a ±1σ population spread).
    """
    ci_df = compute_ci_table(df, metrics=["psnr_db", "ssim", "bpp", "compression_ratio"])
    qs    = sorted(df["quality"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("JPEG DCT — Metrics vs Quality Level  [95 % BCa CI on mean]",
                 fontsize=15, color=TEXT_CLR, fontweight="bold", y=0.98)

    for (col, color, label, ax) in [
        ("psnr_db",           ACCENT1, "PSNR (dB)",            axes[0, 0]),
        ("ssim",              ACCENT4, "SSIM",                  axes[0, 1]),
        ("bpp",               ACCENT3, "BPP",                   axes[1, 0]),
        ("compression_ratio", ACCENT5, "Compression Ratio (×)", axes[1, 1]),
    ]:
        sub   = ci_df[ci_df["metric"] == col].set_index("quality").loc[qs]
        means = sub["mean"].values
        lo    = sub["ci_lo"].values
        hi    = sub["ci_hi"].values

        ax.plot(qs, means, color=color, lw=2.5, marker="o", markersize=8, zorder=4)
        ax.fill_between(qs, lo, hi, color=color, alpha=0.20, label="95 % BCa CI")
        ax.errorbar(qs, means,
                    yerr=[means - lo, hi - means],
                    fmt="none", ecolor=color, elinewidth=1.2,
                    capsize=4, capthick=1.2, alpha=0.7)

        for q, m, h in zip(qs, means, hi):
            ax.annotate(f"{m:.3f}", (q, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=color)

        ax.set_title(label, fontsize=12, color=color, fontweight="bold", pad=8)
        ax.set_xlabel("Quality"); ax.set_ylabel(label)
        ax.set_xticks(qs); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "metrics_vs_quality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show(); print(f"💾  {out}")


def plot_quality_comparison(sample_data: dict, fname: str, df: pd.DataFrame):
    qs   = sorted(sample_data.keys())
    orig = sample_data[qs[0]][0]
    n    = len(qs)
    fig, axes = plt.subplots(3, n+1, figsize=(4*(n+1), 13), facecolor=DARK_BG)
    fig.suptitle(f"JPEG DCT — {fname}",
                 fontsize=14, color=TEXT_CLR, fontweight="bold", y=0.99)
    for row in range(3): axes[row,0].axis("off")
    axes[0,0].imshow(orig); axes[0,0].set_title("ORIGINAL", fontsize=10,
                                                  color=ACCENT1, fontweight="bold")
    axes[0,0].axis("off")
    for ci, q in enumerate(qs, start=1):
        recon = sample_data[q][1]
        diff  = np.abs(orig.astype(int) - recon.astype(int)).astype(np.uint8)
        row_  = df[(df["filename"]==fname) & (df["quality"]==q)].iloc[0]
        col   = Q_COLORS[q]
        axes[0,ci].imshow(recon)
        axes[0,ci].set_title(f"Q={q}", fontsize=10, color=col, fontweight="bold")
        axes[0,ci].axis("off")
        axes[1,ci].imshow(np.clip(diff*4,0,255))
        axes[1,ci].set_title("diff ×4", fontsize=8, color=col); axes[1,ci].axis("off")
        axes[2,ci].axis("off")
        for li, txt in enumerate([f"PSNR  {row_.psnr_db:.2f} dB",
                                   f"SSIM  {row_.ssim:.4f}",
                                   f"BPP   {row_.bpp:.3f}",
                                   f"Ratio {row_.compression_ratio:.2f}×"]):
            axes[2,ci].text(0.5, 0.88-li*0.20, txt, transform=axes[2,ci].transAxes,
                            fontsize=9, color=col, ha="center", fontweight="bold")
    axes[1,0].axis("off"); axes[2,0].axis("off")
    plt.tight_layout(rect=[0,0,1,0.98])
    out = os.path.join(OUTPUT_DIR, "quality_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show(); print(f"💾  {out}")


def plot_rd_comparison(df: pd.DataFrame):
    """
    Rate-distortion curves for all three methods:
      1. Your DCT Codec
      2. Baseline JPEG (cv2, standard Huffman tables)
      3. Optimized JPEG (PIL optimize=True, 2-pass Huffman)
    All plotted at matched BPP so the comparison is fair.
    """
    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle("Rate-Distortion Comparison (avg over dataset) — 3-way",
                 fontsize=14, color=TEXT_CLR, fontweight="bold")

    g = df.groupby("quality").mean(numeric_only=True)

    for ax, metric, ylabel in [
        (ax_psnr, "psnr", "PSNR (dB)"),
        (ax_ssim, "ssim", "SSIM"),
    ]:
        # Your codec
        ax.plot(g["bpp"], g[f"{metric}_db" if metric == "psnr" else metric],
                marker="o", lw=2.5, color=ACCENT1,
                label="Your DCT Codec")

        # Baseline JPEG
        if "jpeg_bpp" in g.columns and g["jpeg_bpp"].sum() > 0:
            ax.plot(g["jpeg_bpp"], g[f"jpeg_{metric}"],
                    marker="s", lw=2.5, color=ACCENT2,
                    label="Baseline JPEG (cv2)")

        # Optimized JPEG
        if "jpeg_opt_bpp" in g.columns and g["jpeg_opt_bpp"].sum() > 0:
            ax.plot(g["jpeg_opt_bpp"], g[f"jpeg_opt_{metric}"],
                    marker="^", lw=2.5, color=ACCENT3,
                    label="Optimized JPEG (PIL optimize=True)")

        ax.set_xlabel("Bits Per Pixel (BPP)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"BPP vs {ylabel}", color=TEXT_CLR)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rd_comparison_3way.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print(f"💾  {out}")


def plot_psnr_delta(df: pd.DataFrame):
    """
    Bar chart: ΔPSNR  =  Your Codec − JPEG variant, per quality level.
    Positive = your codec wins; negative = JPEG variant wins.
    """
    qs = sorted(df["quality"].unique())
    g  = df.groupby("quality").mean(numeric_only=True)

    delta_base = (g["psnr_db"] - g["jpeg_psnr"]).values
    delta_opt  = (g["psnr_db"] - g["jpeg_opt_psnr"]).values

    x  = np.arange(len(qs))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.bar(x - w/2, delta_base, width=w, label="vs Baseline JPEG",  color=ACCENT2, alpha=0.85)
    ax.bar(x + w/2, delta_opt,  width=w, label="vs Optimized JPEG", color=ACCENT3, alpha=0.85)
    ax.axhline(0, color=TEXT_CLR, lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"Q={q}" for q in qs])
    ax.set_ylabel("ΔPSNR (dB)  [Your Codec − JPEG]")
    ax.set_title("PSNR Advantage of Your DCT Codec vs JPEG variants",
                 color=TEXT_CLR, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "psnr_delta.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show(); print(f"💾  {out}")


def plot_confidence_intervals(df: pd.DataFrame):
    """
    Dedicated figure showing BCa 95 % confidence intervals for your DCT
    codec's PSNR and SSIM across quality levels.

    Layout
    ------
    Top row   : PSNR CI bands — one coloured band per quality level,
                plus the distribution of per-image values as a violin.
    Bottom row: SSIM — same structure.

    Reading guide
    -------------
    • The horizontal bar is the 95 % BCa CI on the *mean*.
      Narrower bar = more consistent codec behaviour across images.
    • The violin shows the *population spread* — how much PSNR/SSIM
      varies image-by-image at that quality setting.
    • Non-overlapping bars between quality levels = statistically
      significant difference in mean performance (p < 0.05).
    """
    qs       = sorted(df["quality"].unique())
    q_colors = [Q_COLORS[q] for q in qs]
    ci_psnr  = compute_ci_table(df, metrics=["psnr_db"])
    ci_ssim  = compute_ci_table(df, metrics=["ssim"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle("Your DCT Codec — 95 % BCa Confidence Intervals",
                 fontsize=15, color=TEXT_CLR, fontweight="bold", y=0.98)

    for row_i, (metric, ci_df_m, ylabel) in enumerate([
        ("psnr_db", ci_psnr, "PSNR (dB)"),
        ("ssim",    ci_ssim, "SSIM"),
    ]):
        ax_ci  = axes[row_i, 0]   # horizontal CI bars
        ax_vio = axes[row_i, 1]   # violin / strip

        sub = ci_df_m[ci_df_m["metric"] == metric].set_index("quality").loc[qs]

        # ── CI bar chart ─────────────────────────────────────
        y_pos  = np.arange(len(qs))
        means  = sub["mean"].values
        xerr_lo = means - sub["ci_lo"].values
        xerr_hi = sub["ci_hi"].values - means

        ax_ci.barh(y_pos, xerr_lo + xerr_hi,
                   left=sub["ci_lo"].values,
                   height=0.45, color=q_colors, alpha=0.35, zorder=2)
        ax_ci.scatter(means, y_pos, color=q_colors, s=80, zorder=5,
                      edgecolors=TEXT_CLR, linewidths=0.6)
        ax_ci.errorbar(means, y_pos,
                       xerr=[xerr_lo, xerr_hi],
                       fmt="none", ecolor=TEXT_CLR,
                       elinewidth=1.5, capsize=5, capthick=1.5, zorder=6)

        for i, (q, m, lo, hi, hw) in enumerate(zip(
                qs, means, sub["ci_lo"], sub["ci_hi"], sub["ci_half_width"])):
            ax_ci.text(hi + (hi - lo) * 0.04, i,
                       f"{m:.3f}  ±{hw:.4f}",
                       va="center", ha="left", fontsize=8,
                       color=q_colors[i], fontweight="bold")

        ax_ci.set_yticks(y_pos)
        ax_ci.set_yticklabels([f"Q = {q}" for q in qs], fontsize=10)
        ax_ci.set_xlabel(ylabel); ax_ci.set_title(
            f"{ylabel} — 95 % BCa CI on mean  (n={sub['n'].iloc[0]} images per Q)",
            color=TEXT_CLR, fontsize=10)
        ax_ci.grid(True, alpha=0.25, axis="x")
        ax_ci.invert_yaxis()

        # ── violin + individual points ────────────────────────
        data_per_q = [df[df["quality"] == q][metric].dropna().values for q in qs]
        parts = ax_vio.violinplot(data_per_q, positions=np.arange(len(qs)),
                                  widths=0.6, showmedians=True,
                                  showextrema=False)
        for pc, col in zip(parts["bodies"], q_colors):
            pc.set_facecolor(col); pc.set_alpha(0.35)
        parts["cmedians"].set_color(TEXT_CLR); parts["cmedians"].set_linewidth(1.5)

        # Overlay individual points (jittered)
        for i, (vals, col) in enumerate(zip(data_per_q, q_colors)):
            jitter = _rng.uniform(-0.12, 0.12, size=len(vals))
            ax_vio.scatter(np.full(len(vals), i) + jitter, vals,
                           s=8, alpha=0.25, color=col, zorder=3)

        # Overlay CI bracket
        for i, (lo, hi, col) in enumerate(zip(
                sub["ci_lo"].values, sub["ci_hi"].values, q_colors)):
            ax_vio.plot([i - 0.18, i + 0.18], [lo, lo],
                        color=col, lw=2, solid_capstyle="round")
            ax_vio.plot([i - 0.18, i + 0.18], [hi, hi],
                        color=col, lw=2, solid_capstyle="round")
            ax_vio.plot([i, i], [lo, hi],
                        color=col, lw=1.2, ls="--", alpha=0.7)

        ax_vio.set_xticks(np.arange(len(qs)))
        ax_vio.set_xticklabels([f"Q{q}" for q in qs], fontsize=9)
        ax_vio.set_ylabel(ylabel)
        ax_vio.set_title("Distribution + CI bracket", color=TEXT_CLR, fontsize=10)
        ax_vio.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "confidence_intervals_dct.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show(); print(f"💾  {out}")


def plot_ci_rd_comparison(df: pd.DataFrame):
    """
    Rate-distortion comparison with BCa 95 % CI bands for your DCT codec
    and point estimates (no CI available) for the JPEG variants.
    """
    qs = sorted(df["quality"].unique())
    g  = df.groupby("quality").mean(numeric_only=True)

    # Compute BCa CIs per quality for your codec
    ci_psnr = compute_ci_table(df, metrics=["psnr_db"]).set_index("quality")
    ci_ssim = compute_ci_table(df, metrics=["ssim"]).set_index("quality")
    ci_bpp  = compute_ci_table(df, metrics=["bpp"]).set_index("quality")
    # Re-index to qs
    ci_psnr = ci_psnr[ci_psnr["metric"] == "psnr_db"].loc[qs]
    ci_ssim = ci_ssim[ci_ssim["metric"] == "ssim"].loc[qs]
    ci_bpp  = ci_bpp[ci_bpp["metric"] == "bpp"].loc[qs]

    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(15, 6), facecolor=DARK_BG)
    fig.suptitle("Rate-Distortion — 3-way with 95 % BCa CI (Your Codec)",
                 fontsize=14, color=TEXT_CLR, fontweight="bold")

    for ax, ci_y, y_col, jpeg_y, opt_y, ylabel in [
        (ax_psnr, ci_psnr, "psnr_db", "jpeg_psnr", "jpeg_opt_psnr", "PSNR (dB)"),
        (ax_ssim, ci_ssim, "ssim",    "jpeg_ssim", "jpeg_opt_ssim", "SSIM"),
    ]:
        bpp_mean = ci_bpp["mean"].values
        bpp_lo   = ci_bpp["ci_lo"].values
        bpp_hi   = ci_bpp["ci_hi"].values
        y_mean   = ci_y["mean"].values
        y_lo     = ci_y["ci_lo"].values
        y_hi     = ci_y["ci_hi"].values

        # Your codec — line + 2D CI polygon
        ax.plot(bpp_mean, y_mean, marker="o", lw=2.5, color=ACCENT1,
                label="Your DCT Codec", zorder=5)
        ax.fill_between(bpp_mean, y_lo, y_hi, color=ACCENT1, alpha=0.18,
                        label="95 % BCa CI (PSNR/SSIM)")
        ax.errorbar(bpp_mean, y_mean,
                    xerr=[bpp_mean - bpp_lo, bpp_hi - bpp_mean],
                    fmt="none", ecolor=ACCENT1, elinewidth=0.8,
                    capsize=3, alpha=0.5)

        # Baseline JPEG — point only
        if "jpeg_bpp" in g.columns and g["jpeg_bpp"].sum() > 0:
            ax.plot(g["jpeg_bpp"].values, g[jpeg_y].values,
                    marker="s", lw=2, ls="--", color=ACCENT2,
                    label="Baseline JPEG (cv2)", zorder=4)

        # Optimized JPEG — point only
        if "jpeg_opt_bpp" in g.columns and g["jpeg_opt_bpp"].sum() > 0:
            ax.plot(g["jpeg_opt_bpp"].values, g[opt_y].values,
                    marker="^", lw=2, ls="--", color=ACCENT3,
                    label="Optimized JPEG (PIL)", zorder=4)

        ax.set_xlabel("Bits Per Pixel (BPP)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"BPP vs {ylabel}", color=TEXT_CLR)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rd_comparison_with_ci.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show(); print(f"💾  {out}")


def print_summary(df: pd.DataFrame):
    """Extended summary table with BCa 95 % CI on mean PSNR and SSIM."""
    ci_df = compute_ci_table(df, metrics=["psnr_db", "ssim", "bpp"])

    print("\n" + "═"*120)
    print("  YOUR DCT CODEC — mean ± 95 % BCa CI  |  vs Baseline JPEG  |  vs Optimized JPEG  (matching BPP)")
    print("═"*120)
    hdr = (f"  {'Q':>4}  {'PSNR':>8}  {'CI_PSNR':>17}  {'SSIM':>8}  {'CI_SSIM':>14}  {'BPP':>6}  |  "
           f"{'J_Q':>5} {'J_PSNR':>8} {'J_SSIM':>8}  |  "
           f"{'O_Q':>5} {'O_PSNR':>8} {'O_SSIM':>8}")
    print(hdr)
    print("  " + "─"*120)

    for q in sorted(df["quality"].unique()):
        s = df[df["quality"] == q]
        def m(col): return s[col].mean() if col in s else 0.0

        def _ci(metric):
            row = ci_df[(ci_df["quality"] == q) & (ci_df["metric"] == metric)]
            if row.empty: return "          —"
            lo, hi = row.iloc[0]["ci_lo"], row.iloc[0]["ci_hi"]
            return f"[{lo:.3f}, {hi:.3f}]"

        print(f"  {q:>4}  "
              f"{m('psnr_db'):>8.3f}  {_ci('psnr_db'):>17}  "
              f"{m('ssim'):>8.5f}  {_ci('ssim'):>14}  "
              f"{m('bpp'):>6.3f}  |  "
              f"{m('jpeg_q_match'):>5.1f} {m('jpeg_psnr'):>8.3f} {m('jpeg_ssim'):>8.5f}  |  "
              f"{m('jpeg_opt_q'):>5.1f} {m('jpeg_opt_psnr'):>8.3f} {m('jpeg_opt_ssim'):>8.5f}")

    print("═"*120)
    print("  CI = 95 % BCa bootstrap confidence interval on the mean"
          f"  (n_boot={CI_N_BOOT}, α={CI_ALPHA})\n")

    # Also save CI table to CSV
    ci_all = compute_ci_table(df, metrics=["psnr_db", "ssim", "bpp",
                                            "compression_ratio",
                                            "jpeg_psnr", "jpeg_ssim",
                                            "jpeg_opt_psnr", "jpeg_opt_ssim"])
    ci_path = os.path.join(OUTPUT_DIR, "confidence_intervals.csv")
    ci_all.to_csv(ci_path, index=False)
    print(f"  📄  Full CI table → {ci_path}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, sample_data, sample_name = run_pipeline()
    print_summary(df)                                  # includes BCa CI table + CSV
    plot_quality_comparison(sample_data, sample_name, df)
    plot_metrics_vs_quality(df)                        # BCa CI bands on each metric
    plot_confidence_intervals(df)                      # dedicated CI figure (new)
    plot_rd_comparison(df)                             # 3-way RD curves
    plot_ci_rd_comparison(df)                          # RD curves + CI bands (new)
    plot_psnr_delta(df)
    print("\n🎉  Pipeline complete — 3-way comparison + BCa 95 % CI!")
    print(f"   Outputs in: {OUTPUT_DIR}/")