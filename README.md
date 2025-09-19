# TensLoRA: Tensor Alternatives for Low-Rank Adaptation

Hi!

This repository is the official repository for the paper: **"TensLoRA: Tensor Alternatives for Low-Rank Adaptation,"** submitted at ICASSP2026.

TensLoRA introduces tensor-based alternatives to Low-Rank Adaptation (LoRA) for fine-tuning Transformer models with few parameters.

-----

## What is TensLoRA?

Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning (PEFT) method that adapts large models by adding trainable low-rank matrices to specific layers, typically the attention projections. Standard LoRA, however, treats the updates for each attention projection (Query, Key, Value) and each layer independently, which may create redundancy.

**TensLoRA** extends this idea by aggregating these individual LoRA update matrices into higher-order tensors. By applying tensor factorization techniques, specifically the Tucker factorization, TensLoRA can model the relationships between different dimensions (like attention heads, projection types, and model depth) in a more structured way.

This approach not only generalizes several existing tensor-based methods (like FacT, LoTR, and LoRTA) but also introduces finer control over the parameter budget through mode-specific compression rates.

-----

## Key Features

  * **Unified Framework**: Systematically explores different ways to tensorize LoRA updates.
  * **Flexible Compression**: Allows for mode-specific ranks, enabling tailored parameter allocation based on the task or modality.
  * **State-of-the-Art Generalization**: Captures and extends several existing tensor-based adaptation methods within a single, coherent paradigm.
  * **Competitive Performance**: Experimental results show that certain TensLoRA configurations (e.g., QKV Depth) can outperform standard LoRA with a similar number of parameters.

-----

## Getting Started

### Installation

Clone the repository and install the package as a pip package:

```bash
git clone https://github.com/ax-le/TensLORA.git
cd TensLORA
pip install -e .
```

The required dependencies should be automatically installed.

If you want to exactly reproduce our environments, you could use:

```bash
pip install -r requirements_exact.txt
```

which is the freeze of the environment used in our tests.

### Usage

To reproduce the experiments from the paper, you can run the training scripts. For example:

```bash
# Train a ViT model on the EuroSAT dataset using the 'QKV_Depth' tensor construction
TODO
```

For more details on the available configurations and hyperparameters, please refer to the source code.

-----

## Experimental Results

We evaluated TensLoRA on vision (ViT on EuroSAT, DTD) and language (RoBERTa on COLA, MRPC) benchmarks. Our findings show:

  * The structure of the tensor directly impacts performance.
  * Under an **isoparameters** condition (matching LoRA's parameter count), the **QKV Depth** and **Att QKV Depth** configurations consistently outperform the LoRA baseline.
  * Aggregating along the attention heads dimension (`Att`) was found to be less effective than grouping by projection type (`QKV`) or layer (`Depth`), suggesting redundancy is not uniform across all modes.

These results highlight the potential of tensor-based methods to improve upon LoRA by better modeling structural correlations.

-----

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{marmoret2025tenslora,
  title={{TensLoRA}: Tensor Alternatives for Low-Rank Adaptation},
  author={Marmoret, Axel and Bensaid, Reda and Lys, Jonathan and Gripon, Vincent and Leduc-Primeau, Fran\c{c}ois},
  journal={arXiv preprint arXiv:TODO},
  year={2025}
}
```