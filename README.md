<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/vlm-toolbox-logo.png" alt="VLM Toolbox Logo" width="20%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54" alt="Python Badge">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook Badge">
  <img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge" alt="BSD 3-Clause License">
</p>
<p align="center">
  <em>A deep-learning toolbox for training, fine-tuning, evaluating, and analyzing large vision-language models.</em>
</p>

# Vision-Language Models Toolbox

The **Vision-Language Models Toolbox** is an **all-in-one, flexible** Python library built to **streamline research and development** in multimodal (vision-and-language) learning. Whether you're exploring **soft-prompt finetuning** (e.g., CoOp or CoCoOp) or pushing the boundaries with **large-scale Vision-Language (VL) models** (such as CLIP), this toolbox offers everyth ing you need:

- **📊 Unified** data handling for image, text, and multimodal tasks.  
- **⚙️ Straightforward** model configuration with multiple backbones from OpenAI, Hugging Face, and beyond.  
- **🔄 Extensive** sampling strategies, data **🖼️ augmentation**, and in-depth **📈 evaluation metrics**.  
- **📋 Seamless** logging & visualization with TensorBoard.  
- **🔧 Easy pathways** for integrating **🆕 new models** or **📂 custom datasets** with minimal setup.  
- **🚀 GPU-optimized** training, support for **🔢 half-precision**, sharding, and more.

---

## Table of Contents

- [Vision-Language Models Toolbox](#vision-language-models-toolbox)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Supported Models](#supported-models)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
  - [Adding New Models](#adding-new-models)
  - [Adding a New Dataset](#adding-a-new-dataset)
  - [Notebooks](#notebooks)
  - [Installation](#installation)
    - [1. Conda Environment](#1-conda-environment)
    - [2. Install Dependencies](#2-install-dependencies)
  - [Contributing](#contributing)
  - [License](#license)

---

## Key Features

- **Dataset Handling**: Integrate and preprocess well-known datasets (ImageNet, Food101, Stanford Cars, iNaturalist 2021, MSCOCO Captions, etc.) for both single- and multi-modal tasks.
- **Soft-Prompt Tuning**: Easily leverage CoOp or CoCoOp to adapt large VL models for your specific domain.
- **Advanced Sampling Strategies**: Combat data imbalance with SMOTE, oversampling, undersampling, and more.
- **Data Augmentation**: Effortlessly activate or deactivate standard image/text augmentations to improve model robustness.
- **Metrics & Evaluation**: Track accuracy, top-k, balanced accuracy, F1, AUC, and many other metrics. Compare performance across multiple runs.
- **Logging & Visualization**: Use TensorBoard to monitor model performance, visualize learning curves, and streamline debugging.

---


## Supported Models

| Model Structure       | Provider                                                                                  | Modality   |
|-----------------------|-------------------------------------------------------------------------------------------|------------|
| **CLIP-ViT-B/32**     | [OpenAI](https://openai.com/research/clip)<br>[HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) | Multimodal |
| **CLIP-ViT-B/16**     | [OpenAI](https://openai.com/research/clip)<br>[HuggingFace](https://huggingface.co/openai/clip-vit-base-patch16) | Multimodal |
| **CLIP-ViT-L/14**     | [OpenAI](https://openai.com/research/clip)<br>[HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14) | Multimodal |
| **CLIP-ViT-L/14-336** | [OpenAI](https://openai.com/research/clip)<br>[HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14-336) | Multimodal |
| **CLIP-RN50**         | [OpenAI](https://openai.com/research/clip)                                                                 | Multimodal |
| **CLIP-RN101**        | [OpenAI](https://openai.com/research/clip)                                                                 | Multimodal |
| **CLIP-RN50x4**       | [OpenAI](https://openai.com/research/clip)                                                                 | Multimodal |
| **CLIP-RN50x16**      | [OpenAI](https://openai.com/research/clip)                                                                 | Multimodal |
| **CLIP-RN50x64**      | [OpenAI](https://openai.com/research/clip)                                                                 | Multimodal |
| **DYNO-V2-GIANT**     | [HuggingFace](https://huggingface.co/facebook/dinov2-giant)                                                | Image      |
| **ALL-MiniLM-L6-v2**  | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)                               | Text       |
| **ALL-MPNET-BASE-V2** | [HuggingFace](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)                              | Text       |

---


## Quick Start

**Fine-tuning a CLIP model on ImageNet** is as simple as:

```bash
python src/scripts/train.py \
    --dataset_name imagenet1k \
    --backbone_name vit_b_32 \
    --trainer_name clip \
    --model_type few_shot \
    --setup_type full \
    --num_epochs 100 \
    --train_batch_size 64 \
    --eval_batch_size 256 \
    --precision_dtype fp16 \
    --source huggingface \
    --main_metric_name accuracy \
    --random_state 42 \
    --device_type cuda \
    --collate_all_m2_samples False \
    --save_predictions True
```

This command uses a ViT-B/32 CLIP model from Hugging Face, automatically logs progress, and stores prediction outputs for later review.

---

## Usage

You can also import this toolbox as a library for more advanced or **custom** experimentation. Here’s a minimal code example illustrating how to set up a multimodal pipeline:

```python
from config.enums import (
    CLIPBackbones,
    ImageDatasets,
    Trainers,
    Sources,
    Metrics,
    Stages,
)
from pipeline.pipeline import Pipeline
from config.setup import Setup
from util.memory import flush

# 1. Define your setup
setup = Setup(
    dataset_name=ImageDatasets.IMAGENET_1K,
    backbone_name=CLIPBackbones.CLIP_VIT_B_32,
    trainer_name=Trainers.CLIP,
    model_type='few_shot',
    setup_type='full',
    num_epochs=100,
    train_batch_size=64,
    eval_batch_size=256,
    precision_dtype='fp16',
    source=Sources.HUGGINGFACE,
    main_metric_name=Metrics.ACCURACY,
    random_state=42,
    device_type='cuda'
)

# 2. Initialize the pipeline
pipeline = Pipeline(setup, device_type='cuda')

# 3. Run the training
pipeline.run(
    collate_all_m2_samples=False,
    save_predictions=True,
    persist=True,
)

# 4. Clean up
pipeline.tear_down()
flush()
```

> **Note**: The toolbox treats multiple data inputs as modalities: `m1` and `m2`. This modular design makes it easy to extend to text, image, video, or other data streams.

---

## Adding New Models

One key strength of this repository is **extensibility**. Integrating your own model is straightforward:

1. **Add Your Model to an Enum**  
   Extend `ImageBackbones` or `CLIPBackbones` in [`enums.py`](src/config/enums.py):
   ```python
   class ImageBackbones(BaseEnum):
       DYNO_V2_GIANT = 'dyno_v2_giant'
       NEW_IMAGE_MODEL = 'new_image_model'
   ```

2. **Specify the Model URL**  
   Update [`backbones.py`](src/config/backbones.py):
   ```python
   class BackboneURLConfig(BaseConfig):
       config = {
           Backbones.IMAGE: {
               ImageBackbones.NEW_IMAGE_MODEL: {
                   Sources.HUGGINGFACE: 'new/image-model-url',
               },
           },
           ...
       }
   ```

3. **Train & Evaluate**  
   Reference your model from the command line or from your Python code. Your model is now part of the VL Models Toolbox!

---

## Adding a New Dataset

Similar to adding new models, you can integrate additional datasets seamlessly:

1. **Extend the `ImageDatasets` Enum**  
   In [`enums.py`](src/config/enums.py), add:
   ```python
   class ImageDatasets(BaseEnum):
       IMAGENET_1K = 'imagenet1k'
       FOOD101 = 'food101'
       ...
       MY_NEW_DATASET = 'my_new_dataset'
   ```

2. **Add Configuration**  
   In [`image_datasets.py`](src/config/image_datasets.py), define:
   ```python
   ImageDatasetConfig.config = {
       ...
       ImageDatasets.MY_NEW_DATASET: {
           'splits': ['train', 'validation'],
           DataStatus.RAW: {
               'path': 'HuggingFaceM4/MYNEW',
               'type': StorageType.HUGGING_FACE,
           },
           DataStatus.EMBEDDING: {
               'path': '/path/to/embeddings/my_new_dataset',
               'type': StorageType.DISK,
           },
           'id_col': 'my_label_column_name',
       },
   }
   ```

3. **Validate Paths**  
   If using a local folder, ensure `StorageType.IMAGE_FOLDER` or `StorageType.DISK` is set, and the path exists.

4. **Reference the Dataset**  
   Use `my_new_dataset` in your script or code. That’s it—your dataset is now recognized and processed like any other in the toolbox!

---

## Notebooks

For deeper experimentation and visualization, explore our **Jupyter notebooks** in the [`notebooks`](notebooks) directory:

- **[Zero-Shot Image Classification with CLIP](notebooks/evaluate/zero_shot.ipynb)**  
  Example usage and evaluation for zero-shot scenarios.

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top5-preds-prob.png" alt="Top 5 Predictions Probability" width="80%">
  </p>

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/zero-shot-od.png" alt="Zero-shot Object Detection Model Output" width="80%">
  </p>

- **[Embedding Distribution Visualization](notebooks/analytics/embedding_distribution.ipynb)**  
  Compare embeddings via t-SNE, PCA, and more.

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tsne-all-classes.jpg" alt="VLM Image & Text Embeddings Visualization" width="80%">
  </p>
  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tsne-top-preds.png" alt="Top-k Predictions Image Embedding Visualization" width="80%">
  </p>

- **[Multi-Granular Performance on ImageNet](notebooks/analytics/multi_granular_performance.ipynb)**  
  Assess model accuracy at different class hierarchical levels.

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tree-hierarchy-eval.png" alt="Top-k Predictions Visualization on Label Hierarchy" width="80%">
  </p>

- **[Misclassification Error Analysis](notebooks/analytics/sample_analysis.ipynb)**  
  Gain insights into model misclassifications.

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/gt-heatmap.png" width="80%">
  </p>

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top1-heatmap.png" width="80%">
  </p>

  <p align="center">
      <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top5-heatmap.png" width="80%">
  </p>

---

## Installation

### 1. Conda Environment

```bash
conda create -n vlm-toolbox python=3.9
conda activate vlm-toolbox
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For detailed instructions (e.g., installing separate packages individually), see [ENV_SETUP.md](ENV_SETUP.md).

---

## Contributing

Contributions, suggestions, and new ideas are **highly appreciated**!  
- **Submit Issues & PRs**: If you find bugs or have feature requests, open an [issue](https://github.com/yourusername/vlm-toolbox/issues) or a pull request.  
- **Spread the Word**: Star the repo and share your exciting results to help grow the community.  

For direct inquiries, feel free to reach out:  
**Email**: alirezaheidari dot cs at gmail dot com

---

## License

This project is under the [BSD 3-Clause License](LICENSE).  
Use it freely, modify it, and share your improvements under the same terms.

---

> **Loved This Toolbox?**  
> Give us a ⭐ on GitHub to support the project and help more researchers discover it!  
> **Happy Coding!**
