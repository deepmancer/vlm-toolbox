<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/vision-language-models-toolbox-logo.png" 
       alt="VLM Toolbox Logo" width="80%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54" alt="Python Badge">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook Badge">
  <img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge" alt="BSD 3-Clause License">
</p>

<p align="center">
  <em>A PyTorch-powered library for accelerating multimodal AI research with Vision-Language Models</em>
</p>

# Vision-Language Models Toolbox

A flexible, all-in-one PyTorch library that streamlines research and development with state-of-the-art vision-language models. Whether you’re experimenting with soft-prompt tuning (e.g., CoOp, CoCoOp) or large-scale models such as CLIP, this toolbox provides a robust foundation built on PyTorch and Hugging Face Transformers.

---

## Table of Contents

- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Adding New Models](#adding-new-models)
  - [Adding a New Dataset](#adding-a-new-dataset)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

| **Feature**                 | **Description**                                                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Multimodal Datasets**     | Supports **ImageNet1k, CIFAR-100, Stanford Cars, iNaturalist 2021, MSCOCO Captions**, and more.                                                         |
| **Model Flexibility**       | Works with **CLIP (ViT & ResNet), DINO-V2, MiniLM, MPNet**, and also allows adding custom models.                                                        |
| **Custom Objectives/Tasks** | Quickly add new tasks or losses with minimal code changes for all combined vision-language flows.                                                       |
| **Prompt Tuning**           | Supports **soft prompts (CoOp, CoCoOp) and predefined hard prompts** for dataset adaptation.                                                             |
| **Scalability & Precision** | Supports **multi-GPU, mixed precision (FP16, BF16, FP32, FP64), sharding, and DeepSpeed**.                                                               |
| **Sampling Strategies**     | Includes **oversampling, undersampling, and hybrid methods** like **SMOTE, ADASYN, and Tomek Links**.                                                   |
| **Data Augmentation**       | Provides **image and text augmentations** for model training.                                                                                           |
| **Evaluation Metrics**      | Tracks **accuracy, precision, recall, F1-score, AUC-ROC, and more**.                                                                                    |
| **Logging & Visualization** | Supports **TensorBoard & Loguru** for monitoring and debugging.                                                                                          |
| **Flexible API**            | **Pre-built modules & functionalities** for datasets, models, tasks, setups, and more.                                                                  |

---

## Supported Models

| Backbone           | Supported Provider(s)                                                                                                  | Modality   |
|--------------------|------------------------------------------------------------------------------------------------------------------------|------------|
| **CLIP-ViT-B/32**  | [OpenAI](https://openai.com/research/clip)<br>[Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)       | Multimodal |
| **CLIP-ViT-B/16**  | [OpenAI](https://openai.com/research/clip)<br>[Hugging Face](https://huggingface.co/openai/clip-vit-base-patch16)       | Multimodal |
| **CLIP-ViT-L/14**  | [OpenAI](https://openai.com/research/clip)<br>[Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)       | Multimodal |
| **CLIP-ViT-L/14-336** | [OpenAI](https://openai.com/research/clip)<br>[Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14-336) | Multimodal |
| **CLIP-RN50**      | [OpenAI](https://openai.com/research/clip)                                                                             | Multimodal |
| **CLIP-RN101**     | [OpenAI](https://openai.com/research/clip)                                                                             | Multimodal |
| **CLIP-RN50x4**    | [OpenAI](https://openai.com/research/clip)                                                                             | Multimodal |
| **CLIP-RN50x16**   | [OpenAI](https://openai.com/research/clip)                                                                             | Multimodal |
| **CLIP-RN50x64**   | [OpenAI](https://openai.com/research/clip)                                                                             | Multimodal |
| **DINO-V2-GIANT**  | [Hugging Face](https://huggingface.co/facebook/dinov2-giant)                                                            | Image      |
| **ALL-MiniLM-L6-v2**  | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)                                          | Text       |
| **ALL-MPNET-BASE-V2** | [Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)                                         | Text       |

---

## Quick Start

**Fine-tuning a CLIP model on ImageNet** is as simple as:

```bash
python vlm_toolbox/scripts/train.py \
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

### Running Experiments

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

> **Note**: The toolbox treats multiple data inputs as modalities: `m1` and `m2`. This modular design makes it easy to extend support for text, image, video, or other data streams.

---

### Adding New Models

One key strength of this repository is its **extensibility**. Integrating your own model is straightforward:

1. **Add Your Model to an Enum**  
   Extend `ImageBackbones` or `CLIPBackbones` in 
   [`enums.py`](vlm_toolbox/config/enums.py):
   ```python
   class ImageBackbones(BaseEnum):
       DINO_V2_GIANT = 'dino_v2_giant'
       NEW_IMAGE_MODEL = 'new_image_model'
   ```

2. **Specify the Model URL**  
   Update [`backbones.py`](vlm_toolbox/config/backbones.py):
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
   Reference your new model from the command line or from your Python code. Your model is now part of the VL Models Toolbox!

---

### Adding a New Dataset

Similar to adding new models, you can integrate additional datasets seamlessly:

1. **Extend the `ImageDatasets` Enum**  
   In [`enums.py`](vlm_toolbox/config/enums.py), add:
   ```python
   class ImageDatasets(BaseEnum):
       IMAGENET_1K = 'imagenet1k'
       FOOD101 = 'food101'
       ...
       MY_NEW_DATASET = 'my_new_dataset'
   ```

2. **Add Configuration**  
   In [`image_datasets.py`](vlm_toolbox/config/image_datasets.py), define:
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
   If using a local folder, ensure `StorageType.IMAGE_FOLDER` or `StorageType.DISK` is set, and that the path exists.

4. **Reference the Dataset**  
   Use `my_new_dataset` in your script or code, and you're all set. The dataset is now recognized and processed just like any other!

---

## Jupyter Notebooks

For deeper experimentation and visualization, explore our **Jupyter notebooks** in the [`notebooks`](notebooks) directory:

- **[Zero-Shot Image Classification with CLIP](notebooks/evaluate/zero_shot.ipynb)**  
  Demonstrates example usage and evaluation for zero-shot scenarios.

  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top5-preds-prob.png" 
         alt="Top 5 Predictions Probability" width="50%">
  </p>
  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/zero-shot-od.png" 
         alt="Zero-shot Object Detection Model Output" width="50%">
  </p>

- **[Embedding Distribution Visualization](notebooks/analytics/embedding_distribution.ipynb)**  
  Compare embeddings via t-SNE, PCA, and more.

  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tsne-all-classes.jpg" 
         alt="VLM Image & Text Embeddings Visualization" width="50%">
  </p>
  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tsne-top-preds.png" 
         alt="Top-k Predictions Image Embedding Visualization" width="50%">
  </p>

- **[Multi-Granular Performance on ImageNet](notebooks/analytics/multi_granular_performance.ipynb)**  
  Assess model accuracy at different class hierarchical levels.

  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/tree-hierarchy-eval.png" 
         alt="Top-k Predictions Visualization on Label Hierarchy" width="50%">
  </p>

- **[Misclassification Error Analysis](notebooks/analytics/sample_analysis.ipynb)**  
  Gain insights into where and why the model misclassifies.

  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/gt-heatmap.png" 
         alt="Ground Truth Heatmap" width="50%">
  </p>
  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top1-heatmap.png" 
         alt="Top-1 Prediction Heatmap" width="50%">
  </p>
  <p align="center">
    <img src="https://raw.githubusercontent.com/deepmancer/vlm-toolbox/main/assets/figures/top5-heatmap.png" 
         alt="Top-5 Predictions Heatmap" width="50%">
  </p>

---

## Installation

**1. (Optional) Create a Conda Environment**

```bash
conda create -n vlm python=3.9
conda activate vlm
```

**2. Install From the Source**

```bash
git clone https://github.com/deepmancer/vlm-toolbox.git
cd vlm-toolbox
pip install -e .
```

For more detailed instructions (e.g., installing separate packages individually), see [SETUP.md](SETUP.md).

---

## Acknowledgments

This project benefits from the work of several open-source repositories. We acknowledge and appreciate their contributions to the research community:

- **[OpenAI CLIP](https://github.com/openai/CLIP)**
- **[CoOp](https://github.com/KaiyangZhou/CoOp)**
- **[ProText](https://github.com/muzairkhattak/ProText)**
- **[CuPL](https://github.com/sarahpratt/CuPL)**

---

## Contributing

Contributions, suggestions, and new ideas are **highly appreciated**!

- **Submit Issues & PRs**: If you find bugs or have feature requests, open an [issue](https://github.com/yourusername/vlm-toolbox/issues) or a pull request.  
- **Spread the Word**: Star the repo and share your results to help grow the community.

For direct inquiries, feel free to reach out via email:

**alirezaheidari dot cs at gmail dot com**

---

## License

This project is under the [BSD 3-Clause License](LICENSE).  
Use it freely, modify it, and share your improvements under the same terms.
