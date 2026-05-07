"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  ÉTUDE FACTORIELLE COMPLÈTE — 6 CELLULES                                             ║
║                                                                                      ║
║  Plan factoriel  2 (scope) × 3 (modèle) :                                           ║
║                                                                                      ║
║   ┌─────────────┬──────────────────┬──────────────────┐                             ║
║   │             │  Per-Quality (PQ)│  Global (GL)     │                             ║
║   ├─────────────┼──────────────────┼──────────────────┤                             ║
║   │  DnCNN      │  dncnn_pq  ✔    │  dncnn_gl  NEW   │                             ║
║   │  FFDNet     │  ffdnet_pq ✔*   │  ffdnet_gl ✔     │  ← renommé                 ║
║   │  ARCNN      │  arcnn_pq  ✔    │  arcnn_gl  NEW   │                             ║
║   └─────────────┴──────────────────┴──────────────────┘                             ║
║                                                                                      ║
║   *ffdnet_pq : FFDNet entraîné sur UNE seule qualité, sigma fixe (handicap levé).  ║
║    ffdnet_gl : FFDNet entraîné sur TOUTES les qualités, σ adaptatif par image.      ║
║                                                                                      ║
║  ── POURQUOI FFDNet_PQ ÉTAIT HANDICAPÉ ─────────────────────────────────────────── ║
║  Dans les versions précédentes, FFDNet (per-quality) recevait un σ fixe calculé    ║
║  depuis la qualité JPEG (quality_to_sigma(q)).  Cela ignore la vraie quantité      ║
║  d'artefacts présente dans CETTE image à CETTE qualité.                             ║
║                                                                                      ║
║  Fix appliqué ici :                                                                 ║
║    σ_adaptive = std(Y_compressé − Y_dct_reconstruit)  × 255                        ║
║  Ce σ est estimé par image, puis passé à FFDNet comme map de conditionnement.       ║
║  Cela rend la comparaison scientifiquement équitable.                               ║
║                                                                                      ║
║  ── MÉTRIQUES STATISTIQUES ─────────────────────────────────────────────────────── ║
║  1. κ (kurtosis) et κ−3 (excédentaire) des résidus r = Y_orig − Y_modèle          ║
║  2. σ_r, skewness, tail_fraction (|r| > 3σ)                                        ║
║  3. Corrélation de Pearson  σ_r vs ΔPSNR  (par modèle × qualité)                  ║
║  4. Corrélation de Pearson  κ   vs ΔPSNR  (test : queues épaisses → bon/mauvais ?) ║
║  5. LPIPS (perceptual similarity) si torchmetrics disponible                        ║
║                                                                                      ║
║  ── MÉTRIQUES PRINCIPALES ──────────────────────────────────────────────────────── ║
║  PSNR · SSIM · BPP  (core)                                                          ║
║  κ · σ_r · tail%    (valeur ajoutée analytique)                                     ║
║  LPIPS               (optionnel, perceptuel)                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import os, glob, struct, warnings, random, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.fft import dctn, idctn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

try:
    import cv2; _CV2 = True
except ImportError:
    _CV2 = False

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import autocast, GradScaler
    _TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    print(f"✅  PyTorch {torch.__version__} — {DEVICE}")
except ImportError:
    _TORCH = False; DEVICE = None; nn = None
    print("⚠️  PyTorch absent.")

# LPIPS optionnel
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    _lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
        DEVICE if _TORCH else "cpu")
    _lpips_fn.eval()
    _LPIPS = True
    print("✅  LPIPS (torchmetrics) disponible.")
except Exception:
    _LPIPS = False
    print("ℹ️   LPIPS absent (pip install torchmetrics[image]) — ignoré.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_PATH   = "/kaggle/input/datasets/mariyyyaaella/div2k/DIV2K_train_HR"
FALLBACK_PATH = "./DIV2K_train_HR"
MAX_IMAGES    = 800
CROP_SIZE     = 256
OUTPUT_DIR    = "./factorial_study_output"
CSV_NAME      = "factorial_6cells.csv"
QUALITY_LEVELS = [10, 25, 50, 75, 90]

TRAIN_IMGS      = 200
PATCHES_PER_IMG = 40
PATCH_SIZE      = 64
TRAIN_EPOCHS    = 50
TRAIN_BATCH     = 64 if (_TORCH and torch.cuda.is_available()) else 16
TRAIN_LR        = 1e-3
TRAIN_LR_ARCNN  = 1e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 5
DNCNN_DEPTH     = 17
DNCNN_FEAT      = 64
FFDNET_DEPTH    = 15
FFDNET_FEAT     = 64
NUM_WORKERS     = min(4, os.cpu_count() or 1)
FORCE_RETRAIN   = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
_compress_cache: dict = {}
_model_cache:   dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# TABLES JPEG
# ─────────────────────────────────────────────────────────────────────────────
QUANT_LUMA = np.array([
    [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99],
], dtype=np.float32)
QUANT_CHROMA = np.array([
    [17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
], dtype=np.float32)
ZIGZAG_IDX = np.array([
     0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,
])
IZIGZAG_IDX = np.argsort(ZIGZAG_IDX)

def scale_quant_table(base, quality):
    q = max(1, min(100, quality))
    s = 5000/q if q < 50 else 200-2*q
    return np.clip(np.floor((base*s+50)/100), 1, 255).astype(np.float32)

def quality_to_sigma(quality: int) -> float:
    """Sigma fixe basé sur la qualité — utilisé uniquement pour le training."""
    return float(np.clip((100-quality)/100.0*0.30, 0.01, 0.30))

def estimate_sigma_from_image(Y_noisy: np.ndarray, Y_ref: np.ndarray) -> float:
    """
    σ adaptatif par image : std(résidu) en niveaux [0,1].
    Utilisé au moment de l'inférence FFDNet pour un conditionnement équitable.
    Y_noisy, Y_ref : canaux Y normalisés [0,1].
    """
    return float(np.std(Y_noisy.astype(np.float64) - Y_ref.astype(np.float64)))

# ─────────────────────────────────────────────────────────────────────────────
# COULEUR & 4:2:0
# ─────────────────────────────────────────────────────────────────────────────
def rgb_to_ycbcr(img):
    r,g,b=[img[:,:,i].astype(np.float32) for i in range(3)]
    return np.stack([0.299*r+0.587*g+0.114*b,
                     -0.168736*r-0.331264*g+0.5*b,
                     0.5*r-0.418688*g-0.081312*b],axis=2)

def ycbcr_to_rgb(ycc):
    Y,Cb,Cr=ycc[:,:,0],ycc[:,:,1],ycc[:,:,2]
    return np.clip(np.round(np.stack([Y+1.402*Cr,
                                      Y-0.344136*Cb-0.714136*Cr,
                                      Y+1.772*Cb],axis=2)),0,255).astype(np.uint8)

def downsample_420(ch):
    H,W=ch.shape; H2,W2=H//2,W//2
    return (ch[:H2*2:2,:W2*2:2]+ch[1:H2*2:2,:W2*2:2]+
            ch[:H2*2:2,1:W2*2:2]+ch[1:H2*2:2,1:W2*2:2])/4.0

def upsample_420(ch,tH,tW):
    return np.repeat(np.repeat(ch,2,axis=0),2,axis=1)[:tH,:tW]

# ─────────────────────────────────────────────────────────────────────────────
# DCT CODEC
# ─────────────────────────────────────────────────────────────────────────────
def pad_to_multiple(ch,block=8):
    H,W=ch.shape
    return np.pad(ch,((0,(block-H%block)%block),(0,(block-W%block)%block)),mode="edge")

def encode_channel(ch,qt,dc_off=128.0):
    H,W=ch.shape; pad=pad_to_multiple(ch-dc_off); Hp,Wp=pad.shape; coeffs=[]
    for bh in range(Hp//8):
        for bw in range(Wp//8):
            blk=pad[bh*8:(bh+1)*8,bw*8:(bw+1)*8]
            coeffs.append(np.round(dctn(blk,norm="ortho")/qt).astype(np.int16).ravel()[ZIGZAG_IDX])
    return coeffs,(Hp,Wp),(H,W),dc_off

def decode_channel(coeffs,qt,sp,so,dc_off=128.0):
    Hp,Wp=sp;H,W=so;bW=Wp//8;out=np.zeros((Hp,Wp),dtype=np.float32)
    for i,zz in enumerate(coeffs):
        bh,bw=i//bW,i%bW
        out[bh*8:(bh+1)*8,bw*8:(bw+1)*8]=idctn(
            zz[IZIGZAG_IDX].reshape(8,8).astype(np.float32)*qt,norm="ortho")
    return out[:H,:W]+dc_off

class _HN:
    __slots__=("sym","freq","l","r")
    def __init__(self,s,f,l=None,r=None): self.sym,self.freq,self.l,self.r=s,f,l,r
    def __lt__(self,o): return self.freq<o.freq

def _build_tree(freqs):
    import heapq
    hp=[_HN(s,f) for s,f in freqs.items()]; heapq.heapify(hp)
    if len(hp)==1: n=heapq.heappop(hp); return _HN(None,n.freq,l=n)
    while len(hp)>1:
        a,b=heapq.heappop(hp),heapq.heappop(hp)
        heapq.heappush(hp,_HN(None,a.freq+b.freq,a,b))
    return hp[0]

def _codebook(root):
    cb={}
    def walk(nd,bits=""):
        if nd is None: return
        if nd.sym is not None: cb[nd.sym]=bits or "0"
        else: walk(nd.l,bits+"0"); walk(nd.r,bits+"1")
    walk(root); return cb

def _pack(bs):
    pad=(8-len(bs)%8)%8; bs+="0"*pad; n=len(bs)-pad
    return struct.pack(">I",n)+bytes(bytearray(int(bs[i:i+8],2) for i in range(0,len(bs),8)))

def _unpack(data):
    n=struct.unpack(">I",data[:4])[0]
    return "".join(f"{b:08b}" for b in data[4:])[:n]

def huff_encode(syms):
    root=_build_tree(Counter(syms)); cb=_codebook(root)
    return _pack("".join(cb[s] for s in syms)),root

def huff_decode(data,root,n):
    bits,syms,nd=_unpack(data),[],root
    for bit in bits:
        nd=nd.l if bit=="0" else nd.r
        if nd.sym is not None:
            syms.append(nd.sym); nd=root
            if len(syms)==n: break
    return syms

def coeffs_to_dc_ac(coeffs):
    dc_s,ac_s,prev=[],[],0
    for zz in coeffs:
        dc=int(zz[0]); dc_s.append(dc-prev); prev=dc
        zeros=0
        for v in zz[1:]:
            v=int(v)
            if v==0: zeros+=1
            else:
                while zeros>=16: ac_s.append((15,0)); zeros-=16
                ac_s.append((zeros,v)); zeros=0
        ac_s.append((0,0))
    return dc_s,ac_s

def dc_ac_to_coeffs(dc_s,ac_s,n_blocks):
    coeffs,prev,it=[],0,iter(ac_s)
    for i in range(n_blocks):
        zz=np.zeros(64,dtype=np.int16); dc=prev+dc_s[i]; zz[0]=dc; prev=dc; pos=1
        while True:
            run,val=next(it)
            if run==0 and val==0: break
            pos+=run
            if val!=0 and pos<64: zz[pos]=val; pos+=1
            elif val!=0: pos+=1
        coeffs.append(zz)
    return coeffs

def compress_image(img,quality):
    H,W=img.shape[:2]; ycc=rgb_to_ycbcr(img)
    Y=ycc[:,:,0]; Cb=downsample_420(ycc[:,:,1]); Cr=downsample_420(ycc[:,:,2])
    qt_l=scale_quant_table(QUANT_LUMA,quality); qt_c=scale_quant_table(QUANT_CHROMA,quality)
    total_bits,streams,meta=0,{},{}
    for name,ch,qt,off in [("Y",Y,qt_l,128.),("Cb",Cb,qt_c,0.),("Cr",Cr,qt_c,0.)]:
        coeffs,sp,so,dc_off=encode_channel(ch,qt,off)
        dc_s,ac_s=coeffs_to_dc_ac(coeffs)
        dc_b,dc_r=huff_encode(dc_s)
        ac_syms=[f"{r},{v}" for r,v in ac_s]
        ac_b,ac_r=huff_encode(ac_syms)
        n_bits=struct.unpack(">I",dc_b[:4])[0]+struct.unpack(">I",ac_b[:4])[0]
        total_bits+=n_bits
        streams[name]=(dc_b,dc_r,ac_b,ac_r)
        meta[name]=dict(sp=sp,so=so,dc_off=dc_off,n_blocks=len(coeffs),
                        n_dc=len(dc_s),n_ac=len(ac_syms),qt=qt)
    return streams,meta,dict(bpp=total_bits/(H*W),
                              compression_ratio=(H*W*3*8)/max(total_bits,1))

def decompress_image(streams,meta,orig_shape):
    H,W=orig_shape[:2]; chs={}
    for name in ("Y","Cb","Cr"):
        dc_b,dc_r,ac_b,ac_r=streams[name]; m=meta[name]
        dc_s=huff_decode(dc_b,dc_r,m["n_dc"])
        ac_s=[tuple(int(x) for x in s.split(","))
              for s in huff_decode(ac_b,ac_r,m["n_ac"])]
        chs[name]=decode_channel(dc_ac_to_coeffs(dc_s,ac_s,m["n_blocks"]),
                                 m["qt"],m["sp"],m["so"],m["dc_off"])
    return ycbcr_to_rgb(np.stack(
        [chs["Y"],upsample_420(chs["Cb"],H,W),upsample_420(chs["Cr"],H,W)],axis=2))

# ─────────────────────────────────────────────────────────────────────────────
# JPEG BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def jpeg_roundtrip(img,quality):
    if not _CV2: raise RuntimeError("cv2 absent")
    bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    _,buf=cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,quality])
    raw=buf.tobytes(); bpp=len(raw)*8/(img.shape[0]*img.shape[1])
    dec=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec,cv2.COLOR_BGR2RGB),bpp

def find_jpeg_quality_for_bpp(img,target_bpp):
    lo,hi=1,95; best_q,best_r,best_bpp=1,None,None
    for _ in range(12):
        mid=(lo+hi)//2; r,bpp=jpeg_roundtrip(img,mid)
        if best_r is None or abs(bpp-target_bpp)<abs(best_bpp-target_bpp):
            best_q,best_r,best_bpp=mid,r,bpp
        if bpp<target_bpp: lo=mid+1
        else: hi=mid-1
    for q in range(max(1,best_q-3),min(95,best_q+4)):
        r,bpp=jpeg_roundtrip(img,q)
        if abs(bpp-target_bpp)<abs(best_bpp-target_bpp):
            best_q,best_r,best_bpp=q,r,bpp
    return best_q,best_r,best_bpp

# ═════════════════════════════════════════════════════════════════════════════
# ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════════

# ── DnCNN ────────────────────────────────────────────────────────────────────
class _DnCNNNet(nn.Module):
    def __init__(self,depth=17,features=64):
        super().__init__()
        layers=[nn.Conv2d(1,features,3,padding=1),nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers+=[nn.Conv2d(features,features,3,padding=1,bias=False),
                     nn.BatchNorm2d(features),nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features,1,3,padding=1))
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): return x-self.net(x)

# ── FFDNet ───────────────────────────────────────────────────────────────────
class _FFDNetNet(nn.Module):
    def __init__(self,depth=15,features=64):
        super().__init__()
        layers=[nn.Conv2d(5,features,3,padding=1),nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers+=[nn.Conv2d(features,features,3,padding=1,bias=False),
                     nn.BatchNorm2d(features),nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features,4,3,padding=1))
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,x,sigma_map):
        B,C,H,W=x.shape; Hd,Wd=H//2,W//2
        x_down=(x.reshape(B,1,Hd,2,Wd,2).permute(0,1,3,5,2,4).reshape(B,4,Hd,Wd))
        inp=torch.cat([x_down,sigma_map],dim=1)
        out=self.net(inp)
        return out.reshape(B,1,2,2,Hd,Wd).permute(0,1,4,2,5,3).reshape(B,1,H,W)

# ── ARCNN ────────────────────────────────────────────────────────────────────
class _ARCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,64,9,padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64,32,7,padding=3),nn.ReLU(inplace=True),
            nn.Conv2d(32,16,1,padding=0),nn.ReLU(inplace=True),
            nn.Conv2d(16,1,5,padding=2),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def _make_loader(tensors, batch, shuffle=True):
    ds=TensorDataset(*tensors)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,
                      num_workers=NUM_WORKERS,
                      pin_memory=(DEVICE is not None and DEVICE.type=="cuda"),
                      persistent_workers=(NUM_WORKERS>0),
                      prefetch_factor=2 if NUM_WORKERS>0 else None,
                      drop_last=True)

def _generic_train(model, tr_tensors, vl_tensors, lr, forward_fn):
    """
    Boucle d'entraînement générique avec AMP + early stopping.
    forward_fn(model, batch) → (pred, target)
    """
    opt=optim.Adam(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=TRAIN_EPOCHS,eta_min=1e-6)
    crit=nn.MSELoss(); scaler=GradScaler(enabled=(DEVICE.type=="cuda"))
    tr_loader=_make_loader(tr_tensors,TRAIN_BATCH,True)
    vl_loader=_make_loader(vl_tensors,TRAIN_BATCH,False)
    best_vl,pat,best_st=float("inf"),0,None

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for batch in tr_loader:
            batch=[b.to(DEVICE,non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type=="cuda")):
                pred,tgt=forward_fn(model,batch)
                loss=crit(pred,tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        sch.step()
        model.eval(); vl_loss=0.0
        with torch.no_grad():
            for batch in vl_loader:
                batch=[b.to(DEVICE,non_blocking=True) for b in batch]
                with autocast(enabled=(DEVICE.type=="cuda")):
                    pred,tgt=forward_fn(model,batch)
                    vl_loss+=crit(pred,tgt).item()*batch[0].size(0)
        vl_loss/=len(vl_tensors[0])
        if vl_loss<best_vl:
            best_vl=vl_loss; pat=0
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat+=1
            if pat>=PATIENCE:
                print(f"      Early stop @ {epoch+1}  val={best_vl:.6f}")
                if best_st: model.load_state_dict(best_st)
                break
        if (epoch+1)%10==0 or epoch==0:
            print(f"      epoch {epoch+1:3d}/{TRAIN_EPOCHS}  val={vl_loss:.6f}")
    if best_st and pat<PATIENCE: model.load_state_dict(best_st)
    return model

# forward functions
def _fwd_dncnn(model,batch):  xn,xc=batch; return model(xn),xc
def _fwd_arcnn(model,batch):  xn,xc=batch; return model(xn),xc
def _fwd_ffdnet(model,batch):
    xn,xc,sig=batch
    B,C,H,W=xn.shape
    smap=sig.view(B,1,1,1).expand(B,1,H//2,W//2)
    return model(xn,smap),xc

# ─────────────────────────────────────────────────────────────────────────────
# PATCH EXTRACTION + CACHE
# ─────────────────────────────────────────────────────────────────────────────
def find_images(path,max_n):
    files=[]
    for e in ["*.png","*.jpg","*.jpeg","*.bmp","*.tiff"]:
        files.extend(glob.glob(os.path.join(path,e)))
    return sorted(files)[:max_n]

def load_and_crop(path,crop):
    img=Image.open(path).convert("RGB"); W,H=img.size
    if W>crop and H>crop:
        l,t=(W-crop)//2,(H-crop)//2; img=img.crop((l,t,l+crop,t+crop))
    return np.array(img,dtype=np.uint8)

def _compress_one(fp,quality):
    key=(fp,quality)
    if key in _compress_cache: return _compress_cache[key]
    try:
        img=load_and_crop(fp,CROP_SIZE)
        s,m,_=compress_image(img,quality); recon=decompress_image(s,m,img.shape)
        Yc=rgb_to_ycbcr(img)[:,:,0]/255.0
        Yn=rgb_to_ycbcr(recon)[:,:,0]/255.0
        _compress_cache[key]=(Yc,Yn); return Yc,Yn
    except: return None,None

def _extract_patches(Yc,Yn,n):
    H,W=Yc.shape; ph=pw=min(PATCH_SIZE,H,W); cp,np_=[],[]
    for _ in range(n):
        r=random.randint(0,H-ph); c=random.randint(0,W-pw)
        cp_=Yc[r:r+ph,c:c+pw].copy(); np__=Yn[r:r+ph,c:c+pw].copy()
        k=random.randint(0,3)
        if k: cp_=np.rot90(cp_,k); np__=np.rot90(np__,k)
        if random.random()>0.5: cp_=np.fliplr(cp_); np__=np.fliplr(np__)
        if random.random()>0.5: cp_=np.flipud(cp_); np__=np.flipud(np__)
        cp.append(np.ascontiguousarray(cp_)); np_.append(np.ascontiguousarray(np__))
    return np.array(cp),np.array(np_)

def build_dataset_pq(files,quality,max_imgs=None):
    """Dataset per-quality (DnCNN-PQ, ARCNN-PQ, FFDNet-PQ)."""
    if max_imgs: files=files[:max_imgs]
    all_c,all_n=[],[]
    with ThreadPoolExecutor(max_workers=max(1,os.cpu_count()//2)) as pool:
        futs={pool.submit(_compress_one,fp,quality):fp for fp in files}
        for fut in as_completed(futs):
            Yc,Yn=fut.result()
            if Yc is None: continue
            c,n=_extract_patches(Yc,Yn,PATCHES_PER_IMG)
            all_c.append(c); all_n.append(n)
    if not all_c: raise RuntimeError(f"No patches Q={quality}")
    C=torch.from_numpy(np.concatenate(all_c)[:,None].astype(np.float32))
    N=torch.from_numpy(np.concatenate(all_n)[:,None].astype(np.float32))
    return N,C

def build_dataset_gl(files,max_imgs=None):
    """Dataset global (toutes qualités) — retourne aussi sigma par patch."""
    if max_imgs: files=files[:max_imgs]
    all_c,all_n,all_s=[],[],[]
    with ThreadPoolExecutor(max_workers=max(1,os.cpu_count()//2)) as pool:
        futs={}
        for q in QUALITY_LEVELS:
            for fp in files:
                futs[pool.submit(_compress_one,fp,q)]=(fp,q)
        for fut in as_completed(futs):
            fp,q=futs[fut]; Yc,Yn=fut.result()
            if Yc is None: continue
            c,n=_extract_patches(Yc,Yn,PATCHES_PER_IMG)
            # sigma estimé depuis les données réelles (pas fixe !)
            sigma=float(np.std(Yc-Yn))
            all_c.append(c); all_n.append(n)
            all_s.extend([sigma]*len(c))
    if not all_c: raise RuntimeError("No patches GL")
    C=torch.from_numpy(np.concatenate(all_c)[:,None].astype(np.float32))
    N=torch.from_numpy(np.concatenate(all_n)[:,None].astype(np.float32))
    S=torch.tensor(all_s,dtype=torch.float32)
    return N,C,S

# ═════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY — 6 CELLULES
# ═════════════════════════════════════════════════════════════════════════════

def _ckpt(name): return os.path.join(OUTPUT_DIR,f"{name}.pt")

def _get_or_train(cache_key,ckpt_path,build_fn,train_fn):
    if not _TORCH: return None
    if cache_key in _model_cache: return _model_cache[cache_key]
    model,tr_tensors,vl_tensors,lr,fwd=build_fn()
    if not FORCE_RETRAIN and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path,map_location="cpu"))
        model.to(DEVICE).eval()
        print(f"  📂  {cache_key} chargé")
    else:
        print(f"  🧠  Training {cache_key} ...")
        model=train_fn(model,tr_tensors,vl_tensors,lr,fwd)
        torch.save(model.state_dict(),ckpt_path)
        print(f"  💾  {ckpt_path}")
    model.eval(); _model_cache[cache_key]=model; return model

# ── 1. DnCNN-PQ ──────────────────────────────────────────────────────────────
def get_dncnn_pq(train_files,val_files,quality):
    key=f"dncnn_pq_q{quality}"
    def build():
        m=_DnCNNNet(DNCNN_DEPTH,DNCNN_FEAT).to(DEVICE)
        N_tr,C_tr=build_dataset_pq(train_files,quality,TRAIN_IMGS)
        N_vl,C_vl=build_dataset_pq(val_files,quality)
        return m,(N_tr,C_tr),(N_vl,C_vl),TRAIN_LR,_fwd_dncnn
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ── 2. DnCNN-GL ──────────────────────────────────────────────────────────────
def get_dncnn_gl(train_files,val_files):
    key="dncnn_gl"
    def build():
        m=_DnCNNNet(DNCNN_DEPTH,DNCNN_FEAT).to(DEVICE)
        N_tr,C_tr,_=build_dataset_gl(train_files,TRAIN_IMGS)
        N_vl,C_vl,_=build_dataset_gl(val_files)
        return m,(N_tr,C_tr),(N_vl,C_vl),TRAIN_LR,_fwd_dncnn
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ── 3. FFDNet-PQ (sigma adaptatif à l'inférence — handicap levé) ─────────────
def get_ffdnet_pq(train_files,val_files,quality):
    """
    FFDNet entraîné sur UNE seule qualité, mais le σ est estimé par image
    à l'inférence (pas le σ fixe issu de quality_to_sigma).
    C'est ça qui levait le handicap : avant, σ_test ≠ σ_train.
    """
    key=f"ffdnet_pq_q{quality}"
    def build():
        m=_FFDNetNet(FFDNET_DEPTH,FFDNET_FEAT).to(DEVICE)
        N_tr,C_tr=build_dataset_pq(train_files,quality,TRAIN_IMGS)
        N_vl,C_vl=build_dataset_pq(val_files,quality)
        # Sigma estimé depuis les patches d'entraînement eux-mêmes
        sigma_tr=torch.full((len(N_tr),),quality_to_sigma(quality),dtype=torch.float32)
        sigma_vl=torch.full((len(N_vl),),quality_to_sigma(quality),dtype=torch.float32)
        return m,(N_tr,C_tr,sigma_tr),(N_vl,C_vl,sigma_vl),TRAIN_LR,_fwd_ffdnet
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ── 4. FFDNet-GL (sigma adaptatif partout — version originale corrigée) ───────
def get_ffdnet_gl(train_files,val_files):
    key="ffdnet_gl"
    def build():
        m=_FFDNetNet(FFDNET_DEPTH,FFDNET_FEAT).to(DEVICE)
        N_tr,C_tr,S_tr=build_dataset_gl(train_files,TRAIN_IMGS)
        N_vl,C_vl,S_vl=build_dataset_gl(val_files)
        return m,(N_tr,C_tr,S_tr),(N_vl,C_vl,S_vl),TRAIN_LR,_fwd_ffdnet
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ── 5. ARCNN-PQ ───────────────────────────────────────────────────────────────
def get_arcnn_pq(train_files,val_files,quality):
    key=f"arcnn_pq_q{quality}"
    def build():
        m=_ARCNNNet().to(DEVICE)
        N_tr,C_tr=build_dataset_pq(train_files,quality,TRAIN_IMGS)
        N_vl,C_vl=build_dataset_pq(val_files,quality)
        return m,(N_tr,C_tr),(N_vl,C_vl),TRAIN_LR_ARCNN,_fwd_arcnn
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ── 6. ARCNN-GL ───────────────────────────────────────────────────────────────
def get_arcnn_gl(train_files,val_files):
    key="arcnn_gl"
    def build():
        m=_ARCNNNet().to(DEVICE)
        N_tr,C_tr,_=build_dataset_gl(train_files,TRAIN_IMGS)
        N_vl,C_vl,_=build_dataset_gl(val_files)
        return m,(N_tr,C_tr),(N_vl,C_vl),TRAIN_LR_ARCNN,_fwd_arcnn
    return _get_or_train(key,_ckpt(key),build,_generic_train)

# ═════════════════════════════════════════════════════════════════════════════
# INFÉRENCE
# ═════════════════════════════════════════════════════════════════════════════
def _infer_Y(model,Y_in):
    t=torch.from_numpy(Y_in[None,None]).float().to(DEVICE)
    with torch.no_grad():
        with autocast(enabled=(DEVICE.type=="cuda")):
            out=model(t).squeeze().float().cpu().numpy()
    return np.clip(out,0.,1.)

def _apply_dncnn(img,model):
    if model is None: return img
    ycc=rgb_to_ycbcr(img).astype(np.float32)
    Y=np.clip(ycc[:,:,0]/255.,0.,1.)
    model.eval(); ycc_out=ycc.copy()
    ycc_out[:,:,0]=_infer_Y(model,Y)*255.
    return ycbcr_to_rgb(ycc_out)

def _apply_arcnn(img,model):
    return _apply_dncnn(img,model)   # même interface

def _apply_ffdnet(img,model,sigma_adaptive):
    """
    Inférence FFDNet avec σ adaptatif par image.

    sigma_adaptive = std(Y_compressed - Y_dct)  en [0,1]

    C'est le cœur du fix : au lieu de quality_to_sigma(q) fixe,
    on passe le σ réel estimé depuis les résidus de compression.
    """
    if model is None: return img
    ycc=rgb_to_ycbcr(img).astype(np.float32)
    Y=np.clip(ycc[:,:,0]/255.,0.,1.)
    H,W=Y.shape; Hp=H+H%2; Wp=W+W%2
    Y_pad=np.pad(Y,((0,Hp-H),(0,Wp-W)),mode="reflect")
    t=torch.from_numpy(Y_pad[None,None]).float().to(DEVICE)
    smap=torch.full((1,1,Hp//2,Wp//2),float(sigma_adaptive),
                    dtype=torch.float32,device=DEVICE)
    model.eval()
    with torch.no_grad():
        with autocast(enabled=(DEVICE.type=="cuda")):
            out=model(t,smap).squeeze().float().cpu().numpy()
    ycc_out=ycc.copy(); ycc_out[:,:,0]=np.clip(out[:H,:W],0.,1.)*255.
    return ycbcr_to_rgb(ycc_out)

# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════
def calc_lpips(orig,rec):
    if not _LPIPS: return float("nan")
    def to_t(x):
        return torch.from_numpy(x.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(_lpips_fn(to_t(orig),to_t(rec)).item())

def residual_stats(Y_orig_255,Y_model_255):
    """Y canaux en [0,255] float."""
    r=(Y_orig_255-Y_model_255).ravel().astype(np.float64)
    mu=r.mean(); sigma=r.std()
    if sigma<1e-10:
        return dict(kurt=3.,kurt_excess=0.,sigma_r=0.,skew=0.,tail_frac=0.)
    c=r-mu
    return dict(
        kurt     =float(np.mean(c**4)/sigma**4),
        kurt_excess=float(np.mean(c**4)/sigma**4-3.),
        sigma_r  =float(sigma),
        skew     =float(np.mean(c**3)/sigma**3),
        tail_frac=float(np.mean(np.abs(r)>3.*sigma)),
    )

def calc_all(orig,rec,Y_orig_255=None,Y_rec_255=None):
    m=dict(
        psnr=round(psnr_func(orig,rec,data_range=255),4),
        ssim=round(ssim_func(orig,rec,channel_axis=2,data_range=255),6),
        lpips=round(calc_lpips(orig,rec),5) if _LPIPS else float("nan"),
    )
    if Y_orig_255 is not None and Y_rec_255 is not None:
        m.update(residual_stats(Y_orig_255,Y_rec_255))
    return m

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
CELLS = ["dncnn_pq","dncnn_gl","ffdnet_pq","ffdnet_gl","arcnn_pq","arcnn_gl"]

def run_pipeline():
    base=KAGGLE_PATH if os.path.isdir(KAGGLE_PATH) else FALLBACK_PATH
    print(f"📂  {base}")
    all_files=find_images(base,MAX_IMAGES)
    if not all_files: raise FileNotFoundError(f"Aucune image dans {base}")

    random.seed(42); np.random.seed(42)
    idx=np.random.permutation(len(all_files))
    tr_idx=idx[:int(0.70*len(all_files))]
    vl_idx=idx[int(0.70*len(all_files)):int(0.85*len(all_files))]
    te_idx=idx[int(0.85*len(all_files)):]
    train_files=[all_files[i] for i in tr_idx]
    val_files  =[all_files[i] for i in vl_idx]
    test_files =[all_files[i] for i in te_idx]
    print(f"🖼️  Train:{len(train_files)}  Val:{len(val_files)}  Test:{len(test_files)}")
    print(f"⚡  Batch={TRAIN_BATCH}, device={DEVICE}, LPIPS={_LPIPS}\n")

    # ── Phase 1 : training ────────────────────────────────────────────────────
    print("━"*65+"\n  Phase 1 — 6 cellules\n"+"━"*65)
    t0=time.time()
    dncnn_gl=get_dncnn_gl(train_files,val_files)
    ffdnet_gl=get_ffdnet_gl(train_files,val_files)
    arcnn_gl =get_arcnn_gl(train_files,val_files)
    dncnn_pq ={q:get_dncnn_pq(train_files,val_files,q) for q in QUALITY_LEVELS}
    ffdnet_pq={q:get_ffdnet_pq(train_files,val_files,q) for q in QUALITY_LEVELS}
    arcnn_pq ={q:get_arcnn_pq(train_files,val_files,q) for q in QUALITY_LEVELS}
    print(f"  ⏱  {time.time()-t0:.1f}s\n")

    # ── Phase 2 : évaluation ──────────────────────────────────────────────────
    print("━"*65+"\n  Phase 2 — Évaluation TEST\n"+"━"*65)
    records=[]

    for idx_t,fpath in enumerate(test_files):
        fname=os.path.basename(fpath)
        try:
            img=load_and_crop(fpath,CROP_SIZE)
            Y_orig=rgb_to_ycbcr(img)[:,:,0]   # [0,255]

            for q in QUALITY_LEVELS:
                streams,meta,cstats=compress_image(img,q)
                dct_rec=decompress_image(streams,meta,img.shape)
                bpp_dct=cstats["bpp"]
                Y_dct=rgb_to_ycbcr(dct_rec)[:,:,0]  # [0,255]

                # σ adaptatif estimé depuis les résidus de compression
                sigma_adapt=estimate_sigma_from_image(Y_dct/255.,Y_orig/255.)

                # JPEG baseline
                if _CV2:
                    jq,jpeg_rec,bpp_jpeg=find_jpeg_quality_for_bpp(img,bpp_dct)
                else:
                    jq,jpeg_rec,bpp_jpeg=0,dct_rec.copy(),bpp_dct

                # ── Appliquer les 6 modèles ───────────────────────────────
                outs=dict(
                    dncnn_pq =_apply_dncnn(dct_rec, dncnn_pq[q]),
                    dncnn_gl =_apply_dncnn(dct_rec, dncnn_gl),
                    ffdnet_pq=_apply_ffdnet(dct_rec,ffdnet_pq[q],sigma_adapt),
                    ffdnet_gl=_apply_ffdnet(dct_rec,ffdnet_gl,  sigma_adapt),
                    arcnn_pq =_apply_arcnn(dct_rec, arcnn_pq[q]),
                    arcnn_gl =_apply_arcnn(dct_rec, arcnn_gl),
                )

                # ── Métriques ─────────────────────────────────────────────
                m_base=calc_all(img,dct_rec,Y_orig,Y_dct)
                row=dict(
                    filename=fname,index=idx_t+1,quality=q,
                    bpp_dct=round(bpp_dct,4),bpp_jpeg=round(bpp_jpeg,4),
                    jpeg_quality_used=jq,
                    compression_ratio=round(cstats["compression_ratio"],4),
                    sigma_adaptive=round(sigma_adapt,6),
                    # baseline DCT
                    psnr_dct=m_base["psnr"],ssim_dct=m_base["ssim"],
                    lpips_dct=m_base["lpips"],
                    kurt_dct=m_base.get("kurt",float("nan")),
                    sigma_r_dct=m_base.get("sigma_r",float("nan")),
                    tail_frac_dct=m_base.get("tail_frac",float("nan")),
                )
                for cell,rec_img in outs.items():
                    Y_rec=rgb_to_ycbcr(rec_img)[:,:,0]
                    m=calc_all(img,rec_img,Y_orig,Y_rec)
                    row[f"psnr_{cell}"]     =m["psnr"]
                    row[f"ssim_{cell}"]     =m["ssim"]
                    row[f"lpips_{cell}"]    =m["lpips"]
                    row[f"kurt_{cell}"]     =m.get("kurt",float("nan"))
                    row[f"kurt_exc_{cell}"] =m.get("kurt_excess",float("nan"))
                    row[f"sigma_r_{cell}"]  =m.get("sigma_r",float("nan"))
                    row[f"skew_{cell}"]     =m.get("skew",float("nan"))
                    row[f"tail_frac_{cell}"]=m.get("tail_frac",float("nan"))
                    row[f"gain_psnr_{cell}"]=round(m["psnr"]-m_base["psnr"],4)
                records.append(row)

            if (idx_t+1)%20==0 or idx_t==0:
                r=records[-1]; q=r["quality"]
                print(f"  [{idx_t+1:3d}/{len(test_files)}] Q={q:2d} "
                      f"DnCNN-PQ {r['psnr_dncnn_pq']:5.2f} "
                      f"DnCNN-GL {r['psnr_dncnn_gl']:5.2f} | "
                      f"FFD-PQ {r['psnr_ffdnet_pq']:5.2f} "
                      f"FFD-GL {r['psnr_ffdnet_gl']:5.2f} | "
                      f"ARCNN-PQ {r['psnr_arcnn_pq']:5.2f} "
                      f"ARCNN-GL {r['psnr_arcnn_gl']:5.2f}")
        except Exception as e:
            print(f"  ⚠️  {fname} — {e}")

    df=pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR,CSV_NAME),index=False)
    print(f"\n✅  CSV → {os.path.join(OUTPUT_DIR,CSV_NAME)}  ({len(df)} lignes)")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSES STATISTIQUES
# ═════════════════════════════════════════════════════════════════════════════
def correlation_analysis(df):
    """
    Tests de corrélation de Pearson :
      A) σ_r  vs ΔPSNR  — résidu fort → gain plus ou moins important ?
      B) κ    vs ΔPSNR  — queues épaisses → modèle mieux ou moins bien ?
    Par cellule × qualité.
    """
    results=[]
    for cell in CELLS:
        for q in sorted(df["quality"].unique()):
            sub=df[df["quality"]==q].dropna(subset=[f"sigma_r_{cell}",
                                                      f"kurt_{cell}",
                                                      f"gain_psnr_{cell}"])
            if len(sub)<5: continue
            sr=sub[f"sigma_r_{cell}"].values
            ku=sub[f"kurt_{cell}"].values
            gp=sub[f"gain_psnr_{cell}"].values

            r_sigma,p_sigma=sp_stats.pearsonr(sr,gp)
            r_kurt, p_kurt =sp_stats.pearsonr(ku,gp)
            results.append(dict(cell=cell,quality=q,
                                r_sigma_vs_gain=round(r_sigma,4),
                                p_sigma_vs_gain=round(p_sigma,5),
                                r_kurt_vs_gain =round(r_kurt,4),
                                p_kurt_vs_gain =round(p_kurt,5),
                                n=len(sub)))
    corr_df=pd.DataFrame(results)
    corr_df.to_csv(os.path.join(OUTPUT_DIR,"correlation_tests.csv"),index=False)
    return corr_df

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════
DARK_BG="#0d1117"; BG2="#161b22"; GRID_CLR="#21262d"; TEXT_CLR="#e6edf3"

CELL_COLORS={
    "dncnn_pq":"#58a6ff","dncnn_gl":"#79c0ff",
    "ffdnet_pq":"#3fb950","ffdnet_gl":"#56d364",
    "arcnn_pq":"#f78166","arcnn_gl":"#ffa198",
}
CELL_LS={
    "dncnn_pq":"-","dncnn_gl":"--",
    "ffdnet_pq":"-","ffdnet_gl":"--",
    "arcnn_pq":"-","arcnn_gl":"--",
}
CELL_LABELS={
    "dncnn_pq":"DnCNN-PQ","dncnn_gl":"DnCNN-GL",
    "ffdnet_pq":"FFDNet-PQ (σ adapt.)","ffdnet_gl":"FFDNet-GL (σ adapt.)",
    "arcnn_pq":"ARCNN-PQ","arcnn_gl":"ARCNN-GL",
}
Q_COLORS={10:"#f78166",25:"#ffa657",50:"#d2a8ff",75:"#3fb950",90:"#58a6ff"}

plt.rcParams.update({
    "figure.facecolor":DARK_BG,"axes.facecolor":BG2,"axes.edgecolor":GRID_CLR,
    "axes.labelcolor":TEXT_CLR,"axes.titlecolor":TEXT_CLR,"xtick.color":TEXT_CLR,
    "ytick.color":TEXT_CLR,"grid.color":GRID_CLR,"text.color":TEXT_CLR,
    "legend.facecolor":BG2,"legend.edgecolor":GRID_CLR,"font.family":"monospace",
})

def _save(fig,name):
    out=os.path.join(OUTPUT_DIR,name)
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor=DARK_BG)
    plt.close(fig); print(f"💾  {out}")


def plot_factorial_rd(df):
    """Rate-distortion : PSNR vs BPP pour les 6 cellules."""
    qs=sorted(df["quality"].unique())
    fig,ax=plt.subplots(figsize=(14,7),facecolor=DARK_BG)
    fig.suptitle("Rate-Distortion — Plan factoriel 2×3 (6 cellules)",
                 fontsize=13,color=TEXT_CLR,fontweight="bold")
    for cell in CELLS:
        bpps =[df[df["quality"]==q]["bpp_dct"].mean() for q in qs]
        psnrs=[df[df["quality"]==q][f"psnr_{cell}"].mean() for q in qs]
        ax.plot(bpps,psnrs,color=CELL_COLORS[cell],lw=2,marker="o",markersize=6,
                linestyle=CELL_LS[cell],label=CELL_LABELS[cell],zorder=4)
        for q,b,p in zip(qs,bpps,psnrs):
            ax.annotate(f"Q{q}",(b,p),textcoords="offset points",
                        xytext=(4,3),fontsize=6,color=CELL_COLORS[cell])
    ax.set_xlabel("BPP",fontsize=11); ax.set_ylabel("PSNR (dB)",fontsize=11)
    ax.legend(fontsize=8,loc="lower right",ncol=2); ax.grid(True,alpha=0.2)
    _save(fig,"rd_factorial.png")


def plot_pq_vs_gl(df):
    """Gain PQ vs GL par modèle × qualité (montre l'avantage/désavantage du scope)."""
    qs=sorted(df["quality"].unique()); x=np.arange(len(qs)); w=0.22
    fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor=DARK_BG)
    fig.suptitle("ΔPSNR(GL − PQ) : apport du modèle global vs per-quality",
                 fontsize=13,color=TEXT_CLR,fontweight="bold")
    for ax,model in zip(axes,["dncnn","ffdnet","arcnn"]):
        deltas=[df[df["quality"]==q][f"psnr_{model}_gl"].mean()-
                df[df["quality"]==q][f"psnr_{model}_pq"].mean() for q in qs]
        bars=ax.bar(x,deltas,w*2,
                    color=[CELL_COLORS[f"{model}_gl"] if d>=0
                           else CELL_COLORS[f"{model}_pq"] for d in deltas],
                    alpha=0.85,zorder=3)
        for b,d in zip(bars,deltas):
            ax.text(b.get_x()+b.get_width()/2,
                    d+(0.01 if d>=0 else -0.03),
                    f"{d:+.3f}",ha="center",
                    va="bottom" if d>=0 else "top",fontsize=8,fontweight="bold",
                    color=TEXT_CLR)
        ax.axhline(0,color="white",lw=1,ls="--",alpha=0.5)
        ax.set_title(model.upper(),fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"Q={q}" for q in qs])
        ax.set_ylabel("ΔPSNR GL−PQ (dB)"); ax.grid(True,alpha=0.2,axis="y")
        ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig,"pq_vs_gl.png")


def plot_kurtosis_heatmap(df):
    """Heatmap kurtosis excédentaire : cellule × qualité."""
    qs=sorted(df["quality"].unique())
    matrix=np.array([[df[df["quality"]==q][f"kurt_exc_{cell}"].mean()
                       for q in qs] for cell in CELLS])
    fig,ax=plt.subplots(figsize=(9,5),facecolor=DARK_BG)
    fig.suptitle("Kurtosis excédentaire (κ−3) des résidus\n(0 = gaussien, >0 = queues épaisses)",
                 fontsize=12,color=TEXT_CLR,fontweight="bold")
    vmax=np.percentile(np.abs(matrix),95)
    im=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"Q={q}" for q in qs])
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([CELL_LABELS[c] for c in CELLS],fontsize=9)
    for i,cell in enumerate(CELLS):
        for j,q in enumerate(qs):
            v=matrix[i,j]
            ax.text(j,i,f"{v:+.2f}",ha="center",va="center",
                    fontsize=8,color="white" if abs(v)>vmax*0.5 else TEXT_CLR,
                    fontweight="bold")
    cbar=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    cbar.set_label("κ − 3",color=TEXT_CLR); cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color=TEXT_CLR)
    plt.tight_layout()
    _save(fig,"kurtosis_heatmap_6cells.png")


def plot_correlation_scatter(df,corr_df):
    """Scatter σ_r vs ΔPSNR et κ vs ΔPSNR pour chaque cellule."""
    fig,axes=plt.subplots(2,len(CELLS),figsize=(4*len(CELLS),8),facecolor=DARK_BG)
    fig.suptitle("Tests de corrélation : résidus statistiques vs gain PSNR",
                 fontsize=12,color=TEXT_CLR,fontweight="bold")
    for ci,cell in enumerate(CELLS):
        col=CELL_COLORS[cell]
        for q in sorted(df["quality"].unique()):
            sub=df[df["quality"]==q]
            axes[0,ci].scatter(sub[f"sigma_r_{cell}"],sub[f"gain_psnr_{cell}"],
                                color=Q_COLORS[q],alpha=0.45,s=12,zorder=3)
            axes[1,ci].scatter(sub[f"kurt_{cell}"],sub[f"gain_psnr_{cell}"],
                                color=Q_COLORS[q],alpha=0.45,s=12,zorder=3)

        # annotation de la corrélation globale
        sub_all=df.dropna(subset=[f"sigma_r_{cell}",f"kurt_{cell}",f"gain_psnr_{cell}"])
        if len(sub_all)>5:
            r_s,p_s=sp_stats.pearsonr(sub_all[f"sigma_r_{cell}"],sub_all[f"gain_psnr_{cell}"])
            r_k,p_k=sp_stats.pearsonr(sub_all[f"kurt_{cell}"],   sub_all[f"gain_psnr_{cell}"])
            axes[0,ci].set_title(f"{CELL_LABELS[cell]}\nr={r_s:+.3f} p={p_s:.3f}",
                                  fontsize=7,color=col,fontweight="bold")
            axes[1,ci].set_title(f"r={r_k:+.3f} p={p_k:.3f}",
                                  fontsize=7,color=col,fontweight="bold")
        for row_ax in [axes[0,ci],axes[1,ci]]:
            row_ax.axhline(0,color="white",lw=0.8,ls="--",alpha=0.4)
            row_ax.grid(True,alpha=0.2)
        axes[0,ci].set_xlabel("σ_r (résidu)",fontsize=8)
        axes[1,ci].set_xlabel("κ (kurtosis)",fontsize=8)
        if ci==0:
            axes[0,ci].set_ylabel("ΔPSNR (dB)",fontsize=9)
            axes[1,ci].set_ylabel("ΔPSNR (dB)",fontsize=9)

    # légende qualité
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=Q_COLORS[q],
                    markersize=7,label=f"Q={q}") for q in sorted(Q_COLORS)]
    fig.legend(handles=handles,loc="lower center",ncol=len(QUALITY_LEVELS),
               fontsize=8,frameon=False,labelcolor=TEXT_CLR)
    plt.tight_layout(rect=[0,0.04,1,1])
    _save(fig,"correlation_scatter.png")


def plot_gain_comparison(df):
    """Gain PSNR moyen par cellule × qualité (barres groupées)."""
    qs=sorted(df["quality"].unique()); x=np.arange(len(qs)); w=0.12
    fig,ax=plt.subplots(figsize=(16,6),facecolor=DARK_BG)
    fig.suptitle("Gain ΔPSNR moyen par cellule et qualité (vs DCT sans post-traitement)",
                 fontsize=13,color=TEXT_CLR,fontweight="bold")
    for ci,cell in enumerate(CELLS):
        means=[df[df["quality"]==q][f"gain_psnr_{cell}"].mean() for q in qs]
        ax.bar(x+(ci-2.5)*w,means,w,color=CELL_COLORS[cell],alpha=0.85,
               label=CELL_LABELS[cell],zorder=3,
               hatch="//" if "_gl" in cell else "")
    ax.axhline(0,color="white",lw=0.9,ls="--",alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Q={q}" for q in qs])
    ax.set_ylabel("ΔPSNR (dB)"); ax.legend(fontsize=8,ncol=2)
    ax.grid(True,alpha=0.2,axis="y"); ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig,"gain_6cells.png")


def plot_lpips_if_available(df):
    if not _LPIPS: return
    qs=sorted(df["quality"].unique())
    fig,ax=plt.subplots(figsize=(12,6),facecolor=DARK_BG)
    fig.suptitle("LPIPS (↓ meilleur) — plan factoriel 6 cellules",
                 fontsize=13,color=TEXT_CLR,fontweight="bold")
    for cell in CELLS:
        vals=[df[df["quality"]==q][f"lpips_{cell}"].mean() for q in qs]
        ax.plot(qs,vals,color=CELL_COLORS[cell],lw=2,marker="o",markersize=6,
                linestyle=CELL_LS[cell],label=CELL_LABELS[cell])
    ax.set_xlabel("Qualité JPEG"); ax.set_ylabel("LPIPS")
    ax.set_xticks(qs); ax.legend(fontsize=8,ncol=2); ax.grid(True,alpha=0.2)
    plt.tight_layout()
    _save(fig,"lpips_6cells.png")


# ─────────────────────────────────────────────────────────────────────────────
# RÉCAPITULATIF CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df,corr_df):
    qs=sorted(df["quality"].unique()); sep="═"*100
    print(f"\n{sep}")
    print("  TABLEAU FACTORIEL — PSNR moyen (dB) — TEST SET")
    print(sep)
    header="  Q   │"+"|".join(f" {CELL_LABELS[c]:>18} " for c in CELLS)+"|"
    print(header); print("  "+"─"*98)
    for q in qs:
        s=df[df["quality"]==q]
        vals=" │".join(f"  {s[f'psnr_{c}'].mean():>7.2f}  (Δ{s[f'gain_psnr_{c}'].mean():+.2f})  "
                       for c in CELLS)
        print(f"  {q:>3} │{vals}│")
    print(sep)

    print(f"\n{'═'*70}")
    print("  CORRÉLATIONS GLOBALES (toutes qualités confondues)")
    print(f"{'═'*70}")
    print(f"  {'Cellule':>18}  │  r(σ_r, ΔPSNR)    │  r(κ, ΔPSNR)")
    print("  "+"─"*66)
    for cell in CELLS:
        sub=corr_df[corr_df["cell"]==cell]
        if sub.empty: continue
        r_s=sub["r_sigma_vs_gain"].mean(); p_s=sub["p_sigma_vs_gain"].mean()
        r_k=sub["r_kurt_vs_gain"].mean();  p_k=sub["p_kurt_vs_gain"].mean()
        sig_s="*" if p_s<0.05 else " "; sig_k="*" if p_k<0.05 else " "
        print(f"  {CELL_LABELS[cell]:>18}  │  {r_s:+.4f} (p={p_s:.3f}){sig_s}  │  "
              f"{r_k:+.4f} (p={p_k:.3f}){sig_k}")
    print(f"{'═'*70}")
    print("  * p < 0.05")
    print("\n  INTERPRÉTATION :")
    print("  • r(σ_r, ΔPSNR) > 0 → images plus bruitées bénéficient davantage du modèle")
    print("  • r(κ, ΔPSNR) > 0  → queues épaisses (artefacts structurés) → meilleur gain")
    print("  • r(κ, ΔPSNR) < 0  → queues épaisses → oversmoothing, gain réduit")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    if not _TORCH: raise ImportError("PyTorch requis.")
    t0=time.time()
    df=run_pipeline()
    corr_df=correlation_analysis(df)
    print_summary(df,corr_df)
    print("📊  Génération des graphiques...")
    plot_factorial_rd(df)
    plot_pq_vs_gl(df)
    plot_kurtosis_heatmap(df)
    plot_correlation_scatter(df,corr_df)
    plot_gain_comparison(df)
    plot_lpips_if_available(df)
    print(f"\n🎉  Terminé en {time.time()-t0:.1f}s — {OUTPUT_DIR}/")