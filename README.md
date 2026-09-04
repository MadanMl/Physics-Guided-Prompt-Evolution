# Physics-Guided Prompt Adaptation (ICOP)

**Iterative Correction of Optical Perturbations** — a physics-guided
prompt adaptation framework for optically robust image classification
and object detection.

> Madan M., Reich C., Becker B., Azarhoushang B.  
> *Physics-Guided Prompt Adaptation for Optically Robust Image Classification and Object Detection*  
> Electronics (MDPI), 2026

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
pseudocode for the full detection pipeline is provided in paper.

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
| FashionMNIST | 46.2% | 76.2% | 85.3% | **+9.1 pp** | < 0.001 |
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

@Article{electronics15173985,
AUTHOR = {Madan, Manav and Reich, Christoph and Becker, Björn and Azarhoushang, Bahman},
TITLE = {Physics-Guided Prompt Adaptation for Optically Robust Image Classification and Object Detection},
JOURNAL = {Electronics},
VOLUME = {15},
YEAR = {2026},
NUMBER = {17},
ARTICLE-NUMBER = {3985},
URL = {https://www.mdpi.com/2079-9292/15/17/3985},
ISSN = {2079-9292},
ABSTRACT = {Optical systems in the real world often create image problems, such as Gaussian blur from defocus or atmospheric turbulence, and radial vignetting caused by lens shape. Standard mixed-data fine-tuning helps the task head handle these issues, but it does not actually fix them. We introduce the Iterative Correction of Optical Perturbations ICOP framework, which corrects encoder feature representations before they reach the task head using a physics-aware plug-in module. ICOP does this by modeling blur as an isotropic-Gaussian point-spread function (PSF) and uses gradient-based, self-supervised optimization (Adam) to discover feature-space corrections that steer degraded representations toward their clean-data distribution. It includes a BlurEstimator that builds a degradation descriptor using fixed Laplacian and Sobel operators, and a PromptGenerator that turns this descriptor into modulation parameters for the frozen encoder output. The framework comes in two versions based on the task: an additive correction (ICOP-Add) for image classification, and a Feature-wise Linear Modulation correction (ICOP-FiLM) for object detection. We observe a convergence between clean-task performance and blur-induced degradation across datasets, consistent with greater reliance on high-frequency features in stronger backbones; we treat this as an empirical, cross-dataset observation rather than a demonstrated causal claim (task difficulty, category structure, texture, and object scale also differ across datasets). Independently of this, ICOP-FiLM’s corrective benefit does not scale with degradation severity, revealing a more nuanced relationship between backbone quality and robustness. For classification, ICOP-Add improves distorted-condition accuracy over strong mixed fine-tuning by +2.4, +9.1, and +10.7 percentage points on MNIST, FashionMNIST, and CIFAR-10, respectively (McNemar’s test, p<0.001 on all three, 5000 paired predictions per dataset). On three object detection datasets, ICOP-FiLM improves distorted-condition mAP over a mixed-fine-tuning null hypothesis by +0.026, −0.005, and +0.003 mAP, respectively (all values mean over 3 seeds). Against a matched-blur-ratio control that isolates the correction module’s own contribution, ICOP-FiLM wins by a consistent margin on two of the three datasets (+0.037 and +0.024 mAP, winning in every one of 3/3 seeds on each) and loses on the third (−0.039 mAP, losing in 3/3 seeds); it outperforms parameter-efficient (VPT, Adapter) baselines trained on identical data on the same two datasets. This dataset-dependent pattern is discussed in detail in the main text. ICOP-FiLM adds only 82,672 parameters to a 42-million-parameter Real-Time DEtection TRansformer (RT-DETR) detector. All reported results are obtained under synthetic Gaussian blur and radial vignetting applied to clean images from the six benchmark datasets studied.},
DOI = {10.3390/electronics15173985}
}
```

---

## Object Detection Code

The object detection instantiation (**ICOP-FiLM**) is not publicly
released at this time. It applies a Feature-wise Linear Modulation
(FiLM) correction to the encoder output of RT-DETR via a forward hook,
adding only 82,672 parameters to the 42-million-parameter detector.
The complete training procedure is described in the paper.

---

## License

This code is released for academic and research use. Please cite the
paper if you use this work.
