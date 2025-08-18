# CLaRE: CLIP with Latent Reconstruction Errors for Generated Face Detection

CLaRE is a deepfake detection framework that enhances CLIP by integrating Reconstruction Error (LaRE) `[1]` through a
novel cross-attention mechanism and Context Optimization (CoOp) `[2, 3]` or Conditional Context Optimization (CoCoOp)
`[2, 3]`.

CLaRE integrates CoOp/CoCoOp with CLIP and fuses visual (ViT) and textual features via LaRE and MLP modules.

- **CLIP**: Extracts image and text embeddings  
- **CoOp / CoCoOp**: Learnable textual prompts for classification  
- **LaRE**: Region-level enhancement module  

![CLaRE Pipeline](assets/CLaRE%20Pipeline.png)
![LaRE Extraction Module](assets/LaRE%20Extraction%20Module.png)
![CLaRE Refinement Module](assets/CLaRE%20Refinement%20Module.png)

## Project Structure

* General
    * [`assets/`](assets): Miscellaneous utility files
    * [`bash_scripts/`](scripts): Data downloading scripts
    * [`configs/`](configs): Configuration files for experiments
    * [`core/`](core): Model architecture definitions plus various modules
    * [`jobs/`](jobs): Job scripts for batch execution
    * [`notebooks/`](notebooks): Jupyter notebooks for experimentation/analysis
    * [`src/`](src): Source code for training and testing the pipelines
* Repository
    * `.gitignore`, `environment.yaml`, `README.md`: Repo metadata and dependencies

## Setup and Installation

To set up the environment for this repository, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/2t2c/CLaRE.git
cd CLaRE
```

### 2. Create Environment

```bash
conda env create -f environment.yaml
conda activate clare
```

or using SLURM job

```bash
sbatch run jobs/env_install.job
```

### 2. Weights & Biases (Optional)

Get your key from: [link](https://wandb.ai/authorize)

```bash
wandb login <API Key>
```

### 3. Dataset Downloading & Processing

```bash
# Download DF40 dataset
bash bash_scripts/download_datasets.sh
```

### 4. Run Training and Testing

Config `(configs/config.yaml)` can be modified to adapt to different datasets, categories, and other parameters.

To start the training,

```bash
python main.py --mode train --module fusion --config configs/config.yaml
````

Alternative training pipelines:

```bash
# train with LaRE
python main.py --mode train --module lare --config configs/config.yaml

# train with Clipping (CoOp/CoCoOp)
python main.py --mode train --module clipping --config configs/config.yaml
```

To run evaluation on the test split:

```bash
python main.py --mode test --module fusion --config configs/config.yaml --checkpoint path/to/checkpoint.pth
```

Or using SLURM jobs

```bash
sbatch run jobs/train_fusion.job
sbatch run jobs/test.job
```

## Datasets

We use the [DF40 Dataset](https://github.com/YZY-stack/DF40), which includes:

* **Train**: 31 subsets (~50 Gb)
* **Test**: 40 subsets (~90 Gb)

### Dataset Splits

| **Split**     | **Sources**                                                                                                   |
|---------------|---------------------------------------------------------------------------------------------------------------|
| Training (10) | **Entire Face Synthesis (EFS)**: DiT, SiT, StyleGAN2, StyleGAN3, StyleGANXL, VQGAN, ddim, pixart, rddm, sd2.1 |
| Testing (17)  | **EFS** + **Others** (heygen, MidJourney, whichfaceisreal, stargan, starganv2, styleclip, CollabDiff)         |

## References

* [1] [LaRE2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://github.com/luo3300612/lare)
* [2] [Prompt Learning for Vision-Language Models](https://github.com/KaiyangZhou/CoOp)
* [3] [CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection](https://github.com/sfimediafutures/CLIPping-the-Deception)
