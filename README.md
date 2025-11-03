# Multi-knowledge Driven Heterogeneous Pyramid Model for Enhancing Molecular Property Prediction
## Introduction
PryMol is a novel heterogeneous pyramid framework for molecular property prediction, which contains three main components: Heterogeneous Pyramid Message Passing Neural Network, Multiple Knowledge Enhancement and Fusion Module, and Hierarchical Contrastive Learning Strategy.

<div align=center>
<img src="./PyrMol.jpg" alt="TOC" align=center />
</div>

## Preparation
Clone this repository by " https://github.com/lyj363636/PyrMol.git"
PyrMol is implemented in Pytorch and execute on a one NVIDIA Tesla a100 (40G) GPU. The node and edge features are processed by the open-source package RDKit.

## Environment
Create an environment relied on Python packages:

```python
conda env create -f environment.yml
```

