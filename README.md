# PyrMol: A Knowledge-Structured Pyramid Graph Framework for Generalizable Molecular Property Prediction
## Introduction

<p align="justify">
PryMol is a novel heterogeneous pyramid framework for molecular property prediction, which contains five modules: A. Heterogeneous Pyramid; B. HeterGraph Update; C. Knowledge Fusion; D. Hierarchical Contrast; E.Property Prediction. For a molecular, firstly, the heterogeneous pyramid molecular graph is formalized for Atom-level, Sub-leve, and Mol-level in A module. Then, heterogeneous node updates are performed in B module. Thirly, enhancement and fusion of substructure graphs guided by multi-source knowledge and generate Fuse-embedding in C module, and unifying the embedding dimension of Atom-embedding and Mol-embedding. Fourthly, hierarchical contrastive learning is performed among them in D module. Finally, the features from the three levels are concatenated and fed into the downstream task E. module to predict molecular properties. 
</p>


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
<p align="justify">

To ensure a fair and robust comparison, we have conducted 25 repeated experiments with scaffold splitting using different random seeds. All benchmark datasets have been split as training, validation, and test sets with a ratio of 0.8/0.1/0.1.

We select ten benchmark molecular datasets for experiments including BACE, Blood-brain barrier permeability(BBBP), HIV, ClinTox, Tox21, SIDER, and Toxcast for classification tasks, and ESOL, Freesolv, and Lipophilicity for regression tasks. These data are saved in folder "Datasets_demo.zip".
 
</p>

Firstly,

```sh
unzip Datasets_demo.zip
```

We provide ".csv" files of datasets, which contains "bace", "bbbp", "hiv", "freesolv", "lipophilicity", "eslo","clintox","sider","tox21","toxcast".

For single task datasets, like "bbbp", "bace", "hiv", "freesolv", "lipophilicity", "eslo", use data_slit.py code to generate data.
```sh
python data_slit.py
```

For multiple task datasets, like "clintox","sider","tox21","toxcast", use "multitask_data_slit.py" code to generate data.
```sh
python multitask_data_slit.py
```

And we put data demo files "bace", "clintox" folder in "Datasets_demo", which are data with scaffold splitting using different random seeds.

## Training
Before training codes, you need write the correct data files path in PyrMol/dataset_configs.json.

[And download Mol2Vec file from https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl. Then put this file in same path with main_train.py.]: #

Run the code in PyrMol_demo:

```python
python main_train.py
```

## Models
You can save Model in your file path.
We have uploaded all the data files and checkpoints to [Zenodo](https://markdown.com.cn). And we provide our model demo in "PyrMol/PyrMol_demo/Version3_MultiSub_Contrastive/bace" folder. You can download models in this folder and put them in same path.

