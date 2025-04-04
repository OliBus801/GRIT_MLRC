# README
This repo is the official implementation of [**_Graph Inductive Biases in Transformers without Message Passing_**](https://proceedings.mlr.press/v202/ma23c.html)  (Ma et al., ICML 2023) [[PMLR]](https://proceedings.mlr.press/v202/ma23c.html) [[arXiv]](https://arxiv.org/abs/2305.17589)



> The implementation is based on [GraphGPS (Rampasek et al., 2022)](https://github.com/rampasek/GraphGPS).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/graph-regression-on-zinc-500k)](https://paperswithcode.com/sota/graph-regression-on-zinc-500k?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/graph-classification-on-cifar10-100k)](https://paperswithcode.com/sota/graph-classification-on-cifar10-100k?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/node-classification-on-cluster)](https://paperswithcode.com/sota/node-classification-on-cluster?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/node-classification-on-pattern)](https://paperswithcode.com/sota/node-classification-on-pattern?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/graph-regression-on-zinc-full)](https://paperswithcode.com/sota/graph-regression-on-zinc-full?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/graph-classification-on-peptides-func)](https://paperswithcode.com/sota/graph-classification-on-peptides-func?p=graph-inductive-biases-in-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-inductive-biases-in-transformers/graph-regression-on-peptides-struct)](https://paperswithcode.com/sota/graph-regression-on-peptides-struct?p=graph-inductive-biases-in-transformers)

### Correction of Typos in Paper

There is a typo on $\mathbf{W}\_\text{V}$ in the sentence following Eq. (2). The corrected version is as follws:

``where $\sigma$ is a non-linear activation (ReLU by default); 
$\mathbf{W}\_\text{Q}, \mathbf{W}\_\text{K}, \mathbf{W}\_\text{Ew}, \mathbf{W}\_\text{Eb} \in \mathbb{R}^{d' \times d}$, $\mathbf{W}\_\text{A} \in \mathbb{R}^{1 \times d'}$,  $\mathbf{W}\_\text{V} \in \mathbb{R}^{d \times d}$ and $\mathbf{W}\_\text{Ev} \in \mathbb{R}^{d \times d'}$ are learnable weight matrices; ......''




### Python environment setup with Conda
```bash
conda create -n grit python=3.9
conda activate grit 

# please change the cuda/device version as you need

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --trusted-host download.pytorch.org
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
## conda install openbabel fsspec rdkit -c conda-forge
pip install rdkit

pip install torchmetrics==0.9.1
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym 
pip install pytorch-lightning # required by graphgym 
pip install setuptools==59.5.0
# distuitls has conflicts with pytorch with latest version of setuptools

# ---- experiment management tools --------
# pip install wandb  # the wandb is used in GraphGPS but not used in GRIT (ours); please verify the usability before using.
# pip install mlflow 
### mlflow server --backend-store-uri mlruns --port 5000

```

### Running GRIT
```bash
# Run
python main.py --cfg configs/GRIT/zinc-GRIT.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 2000 seed 41 dataset.dir 'xx/xx/data'

# replace 'cuda:0' with the device to use
# replace 'xx/xx/data' with your data-dir (by default './datasets")
# replace 'configs/GRIT/zinc-GRIT.yaml' with any experiments to run
```

### Configurations and Scripts

- Configurations are available under `./configs/GRIT/xxxxx.yaml`
- Scripts to execute are available under `./scripts/xxx.sh`
  - will run 4 trials of experiments parallelly on `GPU:0,1,2,3`.
 
### Intro to the Code Structure
Our code is based on GraphGym, which intensively relies on the `module registration`. This mechanism allows us to combine modules by module names.
However, it is challenging to trace the code from `main.py`. Therefore, we provide hints for the overall code architecture. 
You can write your customized modules and register them, to build new models under this framework.


The overall architecture of the code: ([x] indicates 'x' is a folder in the code)
```
- model
  - utils
     - [act] (the activation functions: be called by other modules)
     - [pooling] (global pooling functions: be called in output head for graph level tasks)
  - [network] (the macro model architecture: stem->backbone->output head)
  - [encoder] (feature/PE encoders(stem): to bridge inputs to the backbone)
  - [layer] (backbone layer: )
  - [head] (task-dependent output head: )

- training pipeline
  -  data
    - [loader] (data loaders: )
    - [transform] (pre-computed transform: PE and other preprocessing)
  - [train] (training pipeline: logging, visualization, early-stopping, checkpointing, etc.)
  - [optimizer] (optimizers and lr schedulers: )
  - [loss] (loss functions: )

- [config] (the default configurations)
```
#### Notes on RRWP
Storing all RRWP values for large graphs can be memory-intensive, as torch_geometric loads the entire dataset into memory by default. 

Alternatively, you can customize the PyG dataset class or calculate RRWP on the fly within the dataloader. 
Owe to the simplicity of RRWP computations, performing them on the fly only marginally slows down training with multiple processing workers. (for graphs with nodes fewer than 500).

Example config can be found in [cifar10-GRIT-RRWP.yaml](https://github.com/LiamMa/GRIT/blob/f039a252fa8e3bd25103853b831f2d65b4345f6b/configs/GRIT/cifar10-GRIT-RRWP.yaml#L14) (line 5 and line 14).






## Citation
If you find this work useful, please consider citing:

```
@inproceedings{ma2023GraphInductiveBiases,
	title = {Graph {Inductive} {Biases} in {Transformers} without {Message} {Passing}},
	booktitle = {Proc. {Int}. {Conf}. {Mach}. {Learn}.},
	author = {Ma, Liheng and Lin, Chen and Lim, Derek and Romero-Soriano, Adriana and K. Dokania and Coates, Mark and H.S. Torr, Philip and Lim, Ser-Nam},
	year = {2023},
}
```
