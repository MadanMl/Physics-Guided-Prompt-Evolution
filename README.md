# Physics-Guided Prompt Evolution (ICOP)

**Iterative Correction of Optical Perturbations** — a physics-guided
prompt evolution framework for optically robust image classification
and object detection.

> Madan M., Reich C., Becker B., Azarhoushang B.  
> *Physics-Guided Prompt Evolution for Optically Robust Image Classification and Object Detection*  
> Electronics (MDPI), 2025

---

## What is ICOP?

Real-world optical systems introduce two physically predictable
degradations: **isotropic Gaussian blur** from defocus or atmospheric
turbulence, and **radial vignetting** from lens geometry. Standard
mixed-data fine-tuning teaches the model to tolerate the resulting
feature shift — but does not correct it.

ICOP attaches a small correction module to a **frozen, pretrained
encoder** and learns to reverse the blur-induced feature shift in
feature space, before the task head ever sees the corrupted
representation. No labelled degraded data are needed.

The framework has two learned components:

- **BlurEstimator** — measures degradation state using fixed Laplacian
  and Sobel operators (non-learnable, physics-derived) plus a small CNN
- **PromptGenerator** — maps the degradation descriptor to a
  correction applied directly to encoder features

Two task-specific instantiations:

| Variant | Task | Correction type |
|---|---|---|
| **ICOP-Add** | Image classification | Additive: $\hat{f} = f_b + p$ |
| **ICOP-FiLM** | Object detection | FiLM: $\hat{f} = f_b \cdot (1+\gamma) + \beta$ |

---

## Repository Contents

```
.
└── Image_Classification_PhysicsGuidedPromptEvolution.ipynb   ← main notebook
```

This repository currently contains the **image classification
instantiation (ICOP-Add)** as a self-contained Jupyter notebook. The
object detection code (ICOP-FiLM) is not publicly released; the
pseudocode for the full detection pipeline is provided at the bottom of
this README.

---

## Image Classification Notebook

### What it does

The notebook runs the full three-way evaluation on MNIST,
FashionMNIST, and CIFAR-10:

1. **Baseline** — clean-pretrained CNN, evaluated directly on blurry images
2. **FT-Baseline** — same backbone (deepcopy), classifier head fine-tuned on
   mixed data; this is the null hypothesis
3. **ICOP-Add** — same backbone (deepcopy), BlurEstimator + PromptGenerator
   attached and evolved self-supervised, then jointly fine-tuned

The primary metric is **Δ = ICOP blurred accuracy − FT-Baseline blurred
accuracy**, assessed by McNemar's test on 5,000 paired test predictions.

### Results

| Dataset | Baseline (dist.) | FT-Baseline (dist.) | ICOP-Add (dist.) | Δ | p-value |
|---|---|---|---|---|---|
| MNIST | 9.5% | 95.5% | 97.9% | **+2.4 pp** | < 0.001 |
| FashionMNIST | 46.2% | 76.2% | 85.3% | **+9.2 pp** | < 0.001 |
| CIFAR-10 | 22.7% | 46.8% | 57.5% | **+10.7 pp** | < 0.001 |

Evaluation: σ = 1.5, vignette strength s = 0.4. Both ICOP-Add and
FT-Baseline start from the same pretrained backbone with no additional
pre-training for ICOP, ensuring the only difference is the presence of
the correction module.

### Requirements

```bash
pip install torch torchvision tqdm numpy
```

The notebook downloads MNIST, FashionMNIST, and CIFAR-10 automatically
via `torchvision.datasets`. No additional data preparation is needed.

Tested on:
- Python 3.10
- PyTorch 2.x
- CUDA 11.8 / 12.x (CPU also works, significantly slower)

### How to run

Open the notebook in Google Colab or a local Jupyter environment and
run all cells in order. Each dataset (MNIST, FashionMNIST, CIFAR-10)
runs sequentially. Total runtime on a single A100 GPU is approximately
90 minutes for all three datasets.

GPU is strongly recommended for the CIFAR-10 experiments. MNIST and
FashionMNIST complete in reasonable time on CPU.

### Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Blur sigma (σ) | 1.5 | Moderate defocus |
| Vignette strength (s) | 0.4 | Applied during training and eval |
| Pretraining epochs | 60 | Clean data only |
| Evolution iterations | 400 | Self-supervised, no labels |
| Fine-tuning epochs | 30 | Mixed 50/50 data |
| Contrastive margin | 0.3 / 0.4 / 0.6 | Dataset-dependent |
| Dropout | 0.5 | Applied before classifier head |
| Evolution lr | 5×10⁻⁴ | Adam, cosine annealing |

### Training phases

The notebook runs five sequential phases per dataset:

```
Phase 1  →  Pretrain backbone + head on clean data (60 epochs)
             Save checkpoint (φ*, h*)

FT-Baseline  →  deepcopy(φ*); freeze backbone
                Fine-tune head on mixed data (30 epochs)
                [null hypothesis]

Phase 2  →  deepcopy(φ*); attach E and G (zero-init)
             [same checkpoint as FT-Baseline; no additional training]

Phase 3  →  Freeze backbone + head
             Evolve E and G self-supervised (400 iterations)
             Loss = 10·L_corr + 5·L_cos + 2·L_mag + 5·L_sup + 10·L_margin

Phase 4  →  Freeze backbone; unfreeze head + E + G
             Joint fine-tuning on mixed data (30 epochs)
```

### Where the physics enters

- **Degradation synthesis** — the optical simulator applies a Gaussian
  PSF derived from the paraxial diffraction integral, not arbitrary noise
- **BlurEstimator physics branch** — Laplacian and Sobel operators are
  fixed buffers chosen because they directly measure high-frequency
  energy loss, the exact property a Gaussian PSF destroys
- **Correction type** — additive correction compensates the directional
  feature shift that dominates classification sensitivity

---

## Citation

```bibtex
@article{madan2025icop,
  author  = {Madan, Manav and Reich, Christoph and Becker, Bj{\"o}rn
             and Azarhoushang, Bahman},
  title   = {Physics-Guided Prompt Evolution for Optically Robust
             Image Classification and Object Detection},
  journal = {Electronics},
  publisher = {MDPI},
  year    = {2025}
}
```

---

## Object Detection Code

The object detection instantiation (**ICOP-FiLM**) is not publicly
released at this time. It applies a Feature-wise Linear Modulation
(FiLM) correction to the encoder output of RT-DETR via a forward hook,
adding only 82,672 parameters to the 42-million-parameter detector.

The complete training procedure is described by the pseudocode below.
Results on three datasets (Al-Cast, Grinding Wheel, Aquarium) are
reported in the paper.

---

### ICOP-FiLM: Object Detection Pseudocode

```
Algorithm: Object Detection (ICOP-FiLM)
────────────────────────────────────────────────────────────────────

Input:
  D           — clean training dataset
  σ ~ U(0.7, 1.3)   — multi-sigma blur sampling
  m = 0.5     — contrastive margin
  T = 600     — evolution iterations
  E = 30      — fine-tuning epochs

────────────────────────────────────────────────────────────────────
PHASE 1: Pretraining
────────────────────────────────────────────────────────────────────

  Initialise RT-DETR from COCO/Objects365 pre-trained weights
  Train (φ_bb, φ_enc, φ_dec) on clean D for 50 epochs
  Save base model M*
  [M* is reused by all three evaluation models]

────────────────────────────────────────────────────────────────────
PHASE 2: FT-Baseline  (null hypothesis)
────────────────────────────────────────────────────────────────────

  M_ft ← deepcopy(M*)
  Freeze φ_bb, φ_enc  in M_ft
  Fine-tune φ_dec only on mixed data for E epochs
  Evaluate → FT-Baseline mAP

────────────────────────────────────────────────────────────────────
PHASE 3: Evolution
────────────────────────────────────────────────────────────────────

  Attach BlurEstimator E and PromptGenerator G
    (zero-init γ and β output heads) directly to M*
    forming M_icop  [wraps M* in-place; no deepcopy]

  Register FiLM forward hook on φ_enc
  Disable hook  (γ ← None)
  Freeze φ_bb, φ_enc, φ_dec

  for t = 1 to T:

    Sample clean batch x
    σ_t ~ U(0.7, 1.3)
    x̃ ← g_{σ_t} * x                      [blur + vignette, on-the-fly]

    f_c ← MeanPool(φ_enc(x))              [no gradient; hook disabled]
    f_b ← MeanPool(φ_enc(x̃))

    γ_b, β_b ← G(E(x̃))
    f̂_b ← f_b · (1 + γ_b) + β_b         [FiLM on pooled vector]

    γ_c, β_c ← G(E(x))
    p_clean ← concat(γ_c, β_c)
    p_blur  ← concat(γ_b, β_b)

    L = 10·MSE(f_c, f̂_b)                 [L_corr: feature alignment]
      + 5·(1 − cos(f_c, f̂_b))            [L_cos:  angular alignment]
      + 2·MSE(‖f̂_b‖, ‖f_c‖)             [L_mag:  norm matching]
      + 5·mean(p_clean²)                  [L_sup:  suppress on clean]
      + 10·ReLU(‖p_clean‖ − ‖p_blur‖ + m) [L_margin: contrastive]

    Update (E, G) via Adam
      lr = 5×10⁻⁴, cosine annealing, gradient clip 1.0
    Save best (E, G) by L

  Restore best (E*, G*)
  Unfreeze all parameters

────────────────────────────────────────────────────────────────────
PHASE 4: Fine-Tuning
────────────────────────────────────────────────────────────────────

  Freeze φ_bb, φ_enc
  Unfreeze φ_dec, E, G
  Re-enable FiLM hook  [now applied to full (B, S, C) encoder maps]

  lr_dec  = 5×10⁻⁵
  lr_ICOP = 1×10⁻⁴
  Scheduler: OneCycleLR
  Gradient accumulation: 4 steps

  for e = 1 to E:

    Sample mixed batch (50% blurry with jittered σ; 50% clean)

    Forward pass:
      FiLM hook applies  f̂ = f · (1 + γ) + β  to full encoder output

    Update (φ_dec, E, G) via detection loss
      [classification focal loss + L1 box regression + GIoU loss]

    Evaluate val dist-mAP every 5 epochs
    Checkpoint by best dist-mAP

  Return best (φ_dec, E*, G*)

────────────────────────────────────────────────────────────────────
Architecture notes
────────────────────────────────────────────────────────────────────

  BlurEstimator (shared with ICOP-Add):
    Physics branch : Laplacian variance + Sobel gradient variance → R²
    CNN branch     : 3-layer CNN (16ch, BatchNorm, ReLU, stride-2) → R¹⁶
    Fusion         : Linear(18 → 32) + LayerNorm + ReLU
    Total params   : 14,688

  PromptGenerator (ICOP-FiLM):
    Trunk  : Linear(32→64) → GELU → LayerNorm
             Linear(64→128) → GELU → LayerNorm
    Heads  : Linear(128→256) for γ  [zero-init]
             Linear(128→256) for β  [zero-init]
    Total  : 67,984

  ICOP-FiLM total added parameters: 82,672  (<0.2% of 42M RT-DETR)

  Why mean-pool during evolution:
    Computing L_corr on full spatial maps (B, 900, 256) produces
    gradients ~900× larger than on pooled vectors, causing instability.
    Mean-pooling during evolution stabilises training.
    The full spatial FiLM correction is applied at inference.
```

---

## License

This code is released for academic and research use. Please cite the
paper if you use this work.
