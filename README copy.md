# Fine-tuning-Vision-Language-Models

## Env Setup
```bash
conda create --name deep python=3.9.12
conda install pip
pip install --upgrade pip
conda install nodejs # Optional
```

## Conda Token & Solver
```bash
conda install --freeze-installed conda-token
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Base Requirements
```bash
conda install -c nvidia -c conda-forge -c pytorch -c defaults magma-cuda121 astunparse numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses mkl mkl-include
conda install pandas numpy matplotlib seaborn scikit-learn tqdm pre-commit yacs cython

pip install scikit-image imageio plotly dash opencv-python pygraphviz networkx captum ftfy regex nltk
pip install black usort flake8 flake8-bugbear flake8-comprehensions

pip install tensorboard

Usage: tensorboard --logdir=./tmp  --port=8888
```

## Notebook & Lab
```bash
conda install -c conda-forge jupyter notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge jupyterlab
conda install -c miniconda ipykernel nb_conda
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

Usage: jupyter notebook --no-browser --port=8800
```

## Deep Learning Packages
### cudatoolkit
```bash
conda install -c nvidia cuda==12.1.*
```
### pytorch
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### GPU accelerated libs
```bash
conda install faiss-gpu --override-channels -c pytorch
conda install -c rapidsai -c conda-forge -c nvidia rapids=24.02 cuda-version=12.0
```

### transformers
```bash
conda install conda-forge::transformers
conda install -c huggingface -c conda-forge datasets
pip install evaluate
```

### CLIP
```bash
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch
```

### pytorch geometric
```bash
conda install -c conda-forge optuna
conda install pyg -c pyg
conda install pytorch-sparse -c pyg
conda install pytorch-scatter -c pyg
```

### pytorch lightning
```bash
conda install lightning -c conda-forge
```


## Model Interpretability and Visualization
### streamlit
```bash
pip install streamlit
```

### gradio
```bash
pip install gradio
```
### textblob
```bash
pip install textblob
python -m textblob.download_corpora
```

### aif360
```bash
pip install 'aif360[all]'
```

## Vision-Language Frameworks
### lavis
```bash
pip install salesforce-lavis
```
### mmf
```bash
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable . (pip install --editable . --user --no-build-isolation)
```

### pykale
```bash
pip install pykale
```
### openmim
```bash
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
mim install -e ".[multimodal]"
```
### clip-benchmark
```bash
pip install clip-benchmark
```

## Optional Packages
### wandb
```bash
pip install wandb
Usage: wandb login or wandb login --relogin --host=<server url>
```

### kaggle
```bash
pip install kaggle
```

### mlflow
```bash
pip install mlflow
Usage: mlflow server --host 127.0.0.1 --port 8080```
```

### fastai
```bash
conda install -c fastai fastai
```

### UPop
```bash
https://github.com/sdc17/UPop
```






