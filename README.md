# image-compression-research
Training Scope is a Hidden Variable
A Factorial Study of Neural Post-Processing for JPEG Compression
Mariya Derrag — FST Mohammedia, Université Hassan II, Casablanca
What this is
Neural post-processing for JPEG is usually evaluated architecture by architecture. This repository contains the full implementation of a 3×2 factorial study that treats training scope (specialist per-quality vs. generalist global) as an independent variable — not an implementation detail.
What we find
.DnCNN global beats the specialist at Q=50 (+0.43 dB). Cross-quality exposure helps when artifacts are heterogeneous.
.FFDNet degrades under both scopes, even with per-image adaptive σ. The structural mismatch (scalar conditioning + subpixel downsampling vs. block-structured JPEG artifacts) persists regardless of protocol.
.ARCNN global exhibits pathological residual kurtosis runaway at Q=90 (κ−3 up to +6.3). The model hallucinates structure where none remains.
None of these effects are visible when scope is not treated as a variable.
How to reproduce
# 1. Prepare DIV2K (train/val/test split, seed 42)
python data/prepare_div2k.py

# 2. Run corrected codec
python codec/compress.py --quality [10,25,50,75,90]

# 3. Train all 6 cells
python training/train_factorial.py --arch [dncnn|ffdnet|arcnn] --scope [pq|gl]

# 4. Evaluate
python evaluation/evaluate.py --all
