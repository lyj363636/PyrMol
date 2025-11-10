# PyrMol: A Knowledge-Structured Pyramid Graph Framework for Generalizable Molecular Property Prediction
## Introduction
PryMol is a novel heterogeneous pyramid framework for molecular property prediction, which contains five modules: A. Heterogeneous Pyramid; B. HeterGraph Update; C. Knowledge Fusion; D. Hierarchical Contrast; E.Property Prediction. For a molecular, firstly, the heterogeneous pyramid molecular graph is formalized for Atom-level, Sub-leve, and Mol-level in A module. Then, heterogeneous node updates are performed in B module. Thirly, enhancement and fusion of substructure graphs guided by multi-source knowledge and generate Fuse-embedding in C module, and unifying the embedding dimension of Atom-embedding and Mol-embedding. Fourthly, hierarchical contrastive learning is performed among them in D module. Finally, the features from the three levels are concatenated and fed into the downstream task E. module to predict molecular properties. 

<div align=center>
<img src="./PyrMol3.jpg" alt="TOC" align=center />
</div>

## Preparation
Clone this repository by " https://github.com/lyj363636/PyrMol.git"
PyrMol is implemented in Pytorch and execute on a one NVIDIA Tesla a100 (40G) GPU. The node and edge features are processed by the open-source package RDKit.

## Environment
PyrMol primarily relies on the following Python pcakages:
- python=3.12.11
- cuda=12.4
- torch=2.4.0
- numpy=1.26.4
- pandas=2.3.1
- scipy=1.13.1
- tqdm=4.67.1
- scikit-learn=1.7.1
- dgl=2.4.0
  

In case you want to use conda for your own installation please create a new PyrMol environment.
We showed an example of creating an environment.
```sh
conda create -n PyrMol python=3.12.11
conda activate PyrMol
conda install pytorch==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scikit-learn=1.7.1 pandas=2.3.1 numpy=1.26.4 scipy=1.13.1 -c conda-forge
```

Or you can use the provided [environment.yml](./environment.yml) to create all the required dependency packages.
```sh
conda env create -f environment.yml
```

## Data
we select ten benchmark molecular datasets for experiments from MolecularNet[23] including Blood-brain barrier permeability
(BBBP), BACE, HIV, ClinTox, Tox21, SIDER, and Toxcast for classification tasks, and ESOL, Freesolv,
and Lipophilicity for regression tasks. These data are saved in folder "Datasets.zip".

First, you unzip the folder "Datasets.zip"

Then, use code to split the data and save in the folder. We provide an data demo  "bbbp" folder in "Datasets". 
```sh
python data_slit.py
```


## Training
Before training codes, you need write the correct data files path in PyrMol/dataset_configs.json.

Run the codes in PyrMol_demo:

```python
python main_train.py
```

## Models
You can save Model in your file path.
