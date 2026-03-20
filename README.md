# Critical Analysis of SONATA (CVPR 2025)

**Author:** Ruben, Master IASD, Dauphine-PSL  
**Context:** End-of-semester project for the 3D Point Clouds / Perception course.

## Overview

Self-supervised learning on 3D point clouds often suffers from the **Geometric Shortcut**, where networks heavily encode low-level spatial coordinates (like Z-height) rather than learning high-level abstract semantics. 

The recent architecture [Sonata (CVPR 2025)](https://arxiv.org/abs/2312.10008) proposes to reduce this issue using a decoder-free self-distillation approach. 

This repository contains the codebase and the final report for a focused empirical evaluation of Sonata's claims. By designing a custom linear probing and correlation pipeline, we tested whether absolute spatial coordinates could still be linearly recovered from the model's final **dense, up-casted features**.

## Key Findings

Through a focused case study on an unseen indoor scan (`indoor_scan.ply`), we found that:
1. **Strong Spatial Predictability:** The linear probe perfectly reconstructed absolute spatial coordinates from the $1088$-dimensional dense feature space ($R^2 = 0.988$).
2. **Distributed Leakage:** While no single feature acted as a proxy for physical space (Pearson $r \approx 0.219$), the linear combination of the feature ensemble proved that geometric leakage was merely obfuscated rather than eliminated.

Our results highlight the need for careful evaluation of dense feature construction pipelines (like parameter-free up-casting) in 3D SSL.

## Repository Structure

- `report/`: The final LaTeX report (`report.tex`) detailing the mathematical framework, synthetic control validation, and the case study results.
- `experiments/axis1/`: The core Python scripts used to run the feature extraction and the subsequent geometric shortcut probing tests.
- `scripts/`: Bash scripts to reliably launch evaluations on the high-performance computing cluster (MesoNET / Juliet).
- `sonata-article/`: The upstream official Sonata codebase, required for initializing the pre-trained architecture.
- `results/`: Cached logs, feature exports, and zero-shot PCA visual projections.

## Running the Experiments

The evaluation pipeline is built to run on the MesoNET cluster but can be executed locally if the `sonata-article` dependencies are met.

### 1. Synthetic Sanity Check
Validate the diagnostic metrics on a generated, synthetic dataset:
```bash
# This generates the "Shortcut Baseline" and "Semantic Like" representations
bash scripts/run_shortcut_test.sh
```

### 2. Real-Scene Feature Extraction
Extract the dense representations from the Sonata 108M-parameter backbone:
```bash
bash scripts/run_extract_scannet_features.sh
```

### 3. Sonata Geometric Shortcut Evaluation
Evaluate coordinate predictability on the extracted feature payloads:
```bash
MODE=npz INPUT_GLOB='results/features/*.npz' bash scripts/run_shortcut_test.sh
```
