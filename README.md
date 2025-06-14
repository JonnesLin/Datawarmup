# CVPR 2025: From Prototypes to General Distributions  
### An Efficient Curriculum for Masked Image Modeling

**Welcome researchers across vision, audio, and multimodal domains!**  
This curriculum learning library enables **efficient and scalable training** by leveraging sample difficulty, making it especially suited for **large datasets** and **self-supervised learning tasks** like Masked Image Modeling (MIM).

---

## 🔬 Research Applications

Accelerate your work with **data-driven curriculum learning** tailored for:

### 🔍 Vision & Multimodal Tasks
- **Image Generation** (Diffusion, GANs, VAEs): Prioritize simple textures and structures first.
- **Masked Image Modeling** (MAE, SimMIM, BEiT): Train on prototypical samples early, adapt to complex patterns later.
- **Object Detection** (YOLO, DETR, Faster R-CNN): Emphasize clearly defined object instances in early stages.
- **Image Segmentation** (U-Net, Mask R-CNN): Begin with samples featuring distinct, simple boundaries.
- **Video Understanding** (Action recognition, temporal modeling): Start from basic motion sequences.

### 🎧 Audio & Speech
- **Audio Pretraining** (Wav2Vec, HuBERT, WavLM): Focus on low-complexity acoustic patterns first.
- **Speech Recognition** (Whisper, DeepSpeech): Begin with clean speech recordings.
- **Music Generation** (Jukebox, MusicLM): Learn from basic rhythmic and tonal patterns.
- **Audio Classification** (ESC-50, UrbanSound8K): Emphasize distinct and unambiguous sound events.

---

## 📦 What It Does

This library implements the CVPR 2025 approach for **difficulty-based curriculum learning**, using **K-means clustering on pretrained features** to assign progressive training weights:

1. **Feature Extraction**: Extract features using a pretrained model (e.g., DINO).
2. **Clustering**: Cluster similar samples using K-means.
3. **Difficulty Estimation**: Compute distance from cluster centers to estimate difficulty.
4. **Weighted Sampling**: Generate curriculum weights for progressive training.

✅ Helps models learn **from prototypes to complex distributions**, reducing overfitting and accelerating convergence.

---

## 🔧 Key Features

- **Scalable**: Efficient for millions of samples (MiniBatch K-means)
- **Pretrained Support**: Built-in integration with DINO
- **Flexible**: Works with any PyTorch dataset/model
- **Automatic Cluster Selection**: Uses Davies-Bouldin index
- **Progressive Curriculum**: Control warmup and initial sample size
- **Supports Supervised & Unsupervised Modes**
- **Intermediate Layer Feature Extraction**
- **Save/Load Computed Weights**

---

## 🚀 Efficiency Gains

- **Faster Convergence**: Avoid poor local minima
- **Lower Compute**: Focus on most informative samples
- **Automatic Sampling Strategy**: Temperature-controlled difficulty scaling
- **Handles Multi-Modality**: Works across image, audio, text domains

---

## 🧪 Installation

```bash
pip install -r curriculum_requirements.txt
```

---

## 🧭 Quick Example

```python
from curriculum_learning import create_curriculum_learning_with_dino
import torchvision.transforms as transforms

transform = transforms.Compose([...])
dataset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform)
subset = torch.utils.data.Subset(dataset, list(range(100000)))

difficulty_weights, curriculum = create_curriculum_learning_with_dino(
    dataset=subset,
    device='cuda',
    dino_model='dino_vits16',
    batch_size=256,
    initial_effective_percentage=0.15,
    warmup_iterations=5000
)
```

---

## 🔄 Curriculum Integration

```python
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i
        weights = curriculum.get_curriculum_weights(current_iteration=current_iter)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        train_step(data, labels)
```

---

## 🌐 Multi-Domain Usage

```python
# Vision
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=imagenet_dataset, device=device,
    initial_effective_percentage=0.2, warmup_iterations=3000
)

# Audio
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=audio_spectrogram_dataset, device=device,
    initial_effective_percentage=0.12, warmup_iterations=4000
)

# Text
weights, curriculum = create_curriculum_learning(
    model=text_embedding_model, dataset=text_dataset, device=device,
    initial_effective_percentage=0.08, warmup_iterations=8000
)
```

---

## 🧩 API Reference

### `CurriculumLearning`
- `.fit(dataset)`
- `.get_difficulty_weights()`
- `.get_curriculum_weights(current_iteration=N)`
- `.save_weights(filepath)`
- `.load_weights(filepath)`

### `create_curriculum_learning_with_dino()`
- Supports: `dino_vits16`, `dino_vitb16`, `dino_vits8`, `dino_vitb8`

---

## ✅ Best Practices

### Recommended by Dataset Size:
| Size           | Start % | Warmup Iters |
|----------------|---------|--------------|
| <10K           | 20–40%  | 1K–2K        |
| 10K–100K       | 15–25%  | 2K–5K        |
| >100K          | 10–15%  | 5K–10K       |

### Recommended DINO Models:
- `dino_vits16`: Best tradeoff (default)
- `dino_vitb16`: Higher quality, slower
- `dino_vits8`: Faster, lower resolution

### Curriculum Temperature (Automatic):
- No manual tuning needed
- Controlled by `min_temp`, `max_temp` (default: 0.3 → 2.0)
- Adapts from uniform to difficulty-focused sampling

---

## 📈 Performance Tips

- Use GPU for feature extraction
- Larger batches (≥256) improve throughput
- Cache weights to avoid recomputation
- Enable MiniBatchKMeans for >10K datasets
- Leverage curriculum in early training phases

---

## 📄 Citation

```bibtex
@inproceedings{lin2025prototypes,
  title={From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling},
  author={Lin, Jinhong and Wu, Cheng-En and Li, Huanran and Zhang, Jifan and Hu, Yu Hen and Morgado, Pedro},
  booktitle={CVPR},
  year={2025},
  url={https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf}
}
```

📄 [**Read the paper**](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf)
