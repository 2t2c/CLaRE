# Leveraging CLIP-like Models for Deepfake Detection

This project explores multimodal approaches for deepfake detection by integrating visual and textual features using recent methods – LaRE [2], CLIPping [3], and MLP/Attention-based fusion architectures.

## Project Structure

* Modules
  * [`LaRE/`](LaRE): Latent Reconstruction Error
  * [`UFD/`](UFD): UniversalFakeDetect integration
* General 
  * [`configs/`](configs): Configuration files for experiments
  * [`jobs/`](jobs): Job scripts for batch execution
  * [`misc/`](misc): Miscellaneous utility files
  * [`notebooks/`](notebooks): Jupyter notebooks for experimentation/analysis
  * [`results/`](results): Output results and logs
  * [`scripts/`](scripts): Data preprocessing and training bash scripts
  * [`snellius_outputs/`](snellius_outputs): Logs from Snellius HPC cluster
  * [`src/`](src): Source code and model definitions
* Repository
  * `.gitignore`, `environment.yaml`, `README.md`: Repo metadata and dependencies

## Datasets

We use the [DF40 Dataset](https://github.com/YZY-stack/DF40), which includes:

* **Train**: ~50 GB, 31 subsets (real/fake)
* **Test**: 8/40 subsets (real/fake)

## Methodology

The approach integrates CoOp/CoCoOp with CLIP, and fuses visual (ViT) and textual features using LaRE and MLP modules:

## Overall Pipeline

1. **Dataset Preparation**
   Select a subset of DF40 for training/testing (to be finalized).

2. **Training Pipeline**

   * Train **CoOp** on DF40 train split
   * Fuse with **LaRE**, then retrain
     * Enhancement 1: MLP integration
     * Enhancement 2: use RoI pooling
3. **Evaluation Pipeline**
   * Evaluate on DF40 test (8 datasets)
     - heygen
     - MidJourney
     - whichfaceisreal
     - stargan
     - starganv2
     - styleclip
     - e4e
     - CollabDif

### Architecture Components

* **CLIP**: Base feature extractor (image + text embeddings)
* **CoOp/CoCoOp**: Learnable prompts for classification
* **LaRE**: Region-level enhancement (with optional RoI pooling, K/Q/V attention etc.)

**Diagram:**

```
Text (Coop/CoCoOp) ────> [Real/Fake] ────────────┐
                                                 │
ViT Image ───────────────> Img Feat. ───(x)─> Combine ─> Output
    │                                    │
LaRE ──> Stage 1 (*) ──> Stage 2 (**) ───┘

(*) Stage 1:
    - Extracting LaRE representations
(**) Stage 2:
    - MLP Layer
    - LaRE Original Configuration
    - LaRE with RoI Pooling
(x) Shape Matching:
    - Concaenate
    - Summing
```


## References

* [1] [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://github.com/WisconsinAIVision/UniversalFakeDetect)
* [2] [LaRE2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://github.com/luo3300612/lare)
* [3] [CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection](https://github.com/sfimediafutures/CLIPping-the-Deception)
* [4] [DF40: Toward Next-Generation Deepfake Detection](https://github.com/YZY-stack/DF40)
---