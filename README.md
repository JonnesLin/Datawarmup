# Curriculum Learning Library: From Prototypes to General Distributions  
**CVPR 2025** – An Efficient Curriculum for Masked Image Modeling

This PyTorch library implements a **difficulty-based curriculum learning framework** designed for **self-supervised learning at scale**. It enables progressive training by automatically prioritizing simpler samples before gradually including more complex data.

---

## 📌 Overview

This library computes **sample difficulty using pretrained feature representations** (e.g., DINO), clusters the data, and applies **adaptive sampling weights** throughout training.

Key benefits:
- Efficient training on large-scale datasets
- Faster convergence with less overfitting
- Compatible with image, audio, and text domains

---

## 🔬 Use Cases

Supports curriculum learning across multiple domains:

### Vision
- **Masked Image Modeling**: MAE, SimMIM, BEiT
- **Image Generation**: Diffusion models, GANs, VAEs
- **Object Detection**: YOLO, DETR, Faster R-CNN
- **Segmentation**: U-Net, Mask R-CNN
- **Video Understanding**: Action recognition, temporal modeling

### Audio
- **Audio Pretraining**: Wav2Vec, HuBERT, WavLM
- **Speech Recognition**: Whisper, DeepSpeech
- **Music Generation**: Jukebox, MusicLM
- **Audio Classification**: ESC-50, UrbanSound8K

---

## ⚙️ How It Works

1. **Feature Extraction**: Use a pretrained model to obtain feature embeddings.
2. **Clustering**: Apply K-means clustering to group similar samples.
3. **Difficulty Estimation**: Compute each sample's distance to its cluster center.
4. **Weighted Sampling**: Assign training weights based on sample difficulty.

---

## 🔧 Key Features

- Scalable to millions of samples (MiniBatch K-means)
- Compatible with any PyTorch dataset/model
- Pretrained DINO support (`dino_vits16`, `dino_vitb8`, etc.)
- Automatic cluster selection (Davies-Bouldin index)
- Works in supervised and unsupervised settings
- Supports intermediate layer feature extraction
- Save/load difficulty weights for reuse
- Automatic temperature scaling during curriculum

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 🔁 Quick Start

```python
from curriculum_learning import create_curriculum_learning_with_dino
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

difficulty_weights, curriculum = create_curriculum_learning_with_dino(
    dataset=dataset,
    device=device,
    dino_model='dino_vits16',
    batch_size=256,
    initial_effective_percentage=0.15,
    warmup_iterations=5000
)

# Example training loop
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i

        weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=current_iter
        )
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

        # Your training step here
        train_step(inputs, targets)
```

---

## 🧠 API Overview

### `CurriculumLearning`

```python
CurriculumLearning(
    feature_extractor,
    n_clusters=None,
    auto_select_clusters=True,
    cluster_range=(2, 20),
    use_labels=False,
    normalize_weights=True,
    batch_size=256,
    use_minibatch_kmeans=True,
    initial_effective_percentage=0.3,
    warmup_iterations=1000,
    random_state=42
)
```

#### Key Methods
```python
fit(dataset, labels=None, device='cpu')
get_difficulty_weights()
get_curriculum_weights(current_iteration=0, strategy='difficulty')
save_weights(filepath)
load_weights(filepath)
```

---

### `create_curriculum_learning_with_dino()`

Convenience wrapper for DINO-based feature extraction.

```python
create_curriculum_learning_with_dino(
    dataset,
    device,
    dino_model='dino_vits16',
    initial_effective_percentage=0.3,
    warmup_iterations=1000,
    **kwargs
)
```

---

## 🧪 Example: Audio or Text

```python
# Audio spectrogram dataset
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=audio_dataset,
    device=device,
    batch_size=128,
    initial_effective_percentage=0.12,
    warmup_iterations=4000
)

# Text embedding dataset
weights, curriculum = create_curriculum_learning(
    model=text_encoder,
    dataset=text_dataset,
    device=device,
    batch_size=256,
    initial_effective_percentage=0.08,
    warmup_iterations=8000
)
```

---

## 🔬 Automatic Sampling

- Sampling weights are adapted using an internal temperature mechanism.
- Temperature is automatically adjusted to control the **effective dataset size** during training.
- After warmup, the sampler returns **uniform weights** for all samples.

```python
if current_iter < curriculum.warmup_iterations:
    weights = curriculum.get_curriculum_weights(current_iteration=current_iter)
else:
    weights = [1.0 / len(dataset)] * len(dataset)  # Uniform sampling
```

---

## 📄 Citation

```bibtex
@inproceedings{lin2025prototypes,
  title={From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling},
  author={Lin, Jinhong and Wu, Cheng-En and Li, Huanran and Zhang, Jifan and Hu, Yu Hen and Morgado, Pedro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  url={https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf}
}
```

📖 [Read the Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf)

---
