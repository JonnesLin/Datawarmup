# CVPR 2025: From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling

**🚀 Welcome researchers from diverse fields!** This curriculum learning library is designed to accelerate training across various domains:

### 🌟 Research Applications

**Unlock the power of intelligent sample selection for your research!** This curriculum learning library supports:

#### Vision & Multimodal Research
- **🖼️ Image Generation**: Diffusion models, GANs, VAEs - focus on simple patterns first
- **🎭 Masked Image Modeling**: MAE, SimMIM, BEiT - learn from prototypical examples
- **🔍 Object Detection**: YOLO, R-CNN, DETR - start with clear, well-defined objects
- **✂️ Image Segmentation**: U-Net, Mask R-CNN - progress from simple to complex boundaries
- **🎥 Video Understanding**: Action recognition, temporal modeling - build from basic motions

#### Audio & Speech Research  
- **🎵 Audio Pretraining**: Wav2Vec, HuBERT, WavLM - learn fundamental audio patterns first
- **🗣️ Speech Recognition**: Whisper, DeepSpeech - start with clear pronunciation
- **🎼 Music Generation**: Jukebox, MusicLM - understand basic musical structures
- **🔊 Audio Classification**: Environmental sounds, music genres - recognize distinct patterns

#### Natural Language Processing
- **🔤 Tokenizer Training**: BPE, SentencePiece, WordPiece - learn common patterns first
- **📚 Language Model Pretraining**: BERT, GPT, T5 - build from simple to complex language
- **🌐 Machine Translation**: Transformer models - start with direct word mappings
- **📝 Text Classification**: BERT-based models - learn clear category distinctions

#### Specialized Domains
- **🎮 Reinforcement Learning**: Experience replay prioritization
- **🕸️ Graph Neural Networks**: Node/edge difficulty assessment  
- **📈 Time Series**: Sequence complexity modeling
- **🏥 Medical Imaging**: Pathology detection, medical classification

---

A ready-to-use curriculum learning library that uses K-means clustering on pretrained model features to generate difficulty-based sample weights for **efficient large-scale training**. 

**🚀 Accelerate your training with intelligent sample selection - achieve better results with less computational cost!**

**Based on the CVPR 2025 paper: "From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling"**

## 🎯 Large-Scale Training Efficiency

This library is specifically designed to **accelerate large-scale training** by:
- **Reducing training time**: Focus on the most informative samples first
- **Improving convergence**: Structured learning progression prevents getting stuck in local minima  
- **Scaling efficiently**: Optimized for datasets with millions of samples
- **Memory optimization**: Intelligent batching and MiniBatch K-means for large datasets
- **GPU acceleration**: Fully optimized for multi-GPU training workflows

## Overview

This library implements a curriculum learning approach inspired by our CVPR 2025 research on improving Masked Image Modeling (MIM) through prototype-driven curriculum learning. The core methodology:

1. **Feature Extraction**: Features are extracted from your data using any pretrained model (DINO recommended)
2. **Clustering**: K-means clustering groups similar samples together
3. **Difficulty Assessment**: Samples farther from cluster centers are considered more difficult
4. **Curriculum Weights**: Generated weights enable progressive training from easy to hard samples

Our approach addresses the fundamental challenge in MIM where models are expected to learn complex distributions from partial observations before developing basic capabilities. By structuring the learning process to progress from prototypical examples to complex variations, we achieve more efficient and stable learning trajectories.

## Key Features

- ✅ **Large-Scale Optimized**: Handles millions of samples with efficient memory management
- ✅ **DINO Integration**: Default support for DINO models (recommended for best performance)
- ✅ **Universal Compatibility**: Works with any PyTorch model and dataset
- ✅ **Auto-Cluster Selection**: Automatically finds optimal number of clusters using Davies-Bouldin index
- ✅ **Curriculum Control**: Configurable initial dataset percentage and warmup iterations
- ✅ **Supervised & Unsupervised**: Supports both clustering approaches
- ✅ **Memory Efficient**: Handles large datasets with batched processing
- ✅ **Intermediate Features**: Extract features from any model layer
- ✅ **Save/Load Weights**: Persist computed weights for reuse

## Installation

```bash
pip install -r curriculum_requirements.txt
```

## Quick Start

### Basic Usage with Curriculum Progression

```python
import torch
import torchvision
from curriculum_learning import create_curriculum_learning_with_dino

# Load your dataset (ImageNet example for large-scale training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)

# Use meaningful subset for demonstration
subset_indices = list(range(100000))  # 100k samples for large-scale training
dataset = torch.utils.data.Subset(dataset, subset_indices)

# Generate curriculum weights using DINO (recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
difficulty_weights, curriculum = create_curriculum_learning_with_dino(
    dataset=dataset,
    device=device,
    dino_model='dino_vits16',  # Default DINO variant
    batch_size=256,  # Large batch for efficiency
    initial_effective_percentage=0.15,  # Start with 15% of dataset
    warmup_iterations=5000  # Warmup for 5000 iterations
)

# Training loop with curriculum progression (temperature computed automatically)
for epoch in range(num_epochs):
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i
        
        # Get curriculum weights for current iteration (temperature automatic)
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=current_iter
        )
        
        # Update sampler with new weights
        sampler = WeightedRandomSampler(curriculum_weights, len(dataset), replacement=True)
        
        # Your training code here
        train_step(batch_data, batch_labels)
```

### Multi-Domain Applications

```python
# Image Generation Example (ImageNet-scale)
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=imagenet_dataset,
    device=device,
    batch_size=128,
    initial_effective_percentage=0.2,  # Start with prototypical images
    warmup_iterations=3000
)

# Audio Pretraining Example (Large-scale)
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=audio_spectrogram_dataset,
    device=device,
    batch_size=128,
    initial_effective_percentage=0.12,  # Start with simpler audio patterns
    warmup_iterations=4000  # Longer warmup for temporal patterns
)

# Tokenizer Training Example (Large-scale)
weights, curriculum = create_curriculum_learning(
    model=text_embedding_model,
    dataset=text_dataset, 
    device=device,
    batch_size=256,
    initial_effective_percentage=0.08,  # Start with common patterns
    warmup_iterations=8000  # Long warmup for linguistic complexity
)
```

## API Reference

### CurriculumLearning

The main class for curriculum learning.

#### Parameters

- `feature_extractor`: FeatureExtractor instance
- `n_clusters`: Number of clusters (None for auto-selection)
- `auto_select_clusters`: Whether to auto-select optimal clusters
- `cluster_range`: Range for auto-selection (default: (2, 20))
- `use_labels`: Use supervised clustering (default: False)
- `normalize_weights`: Normalize difficulty weights (default: True)
- `batch_size`: Batch size for processing (default: 256)
- `use_minibatch_kmeans`: Use MiniBatchKMeans for large datasets (default: True)
- `initial_effective_percentage`: Initial percentage of dataset to use (default: 0.3)
- `warmup_iterations`: Number of warmup iterations for curriculum progression (default: 1000)
- `random_state`: Random seed (default: 42)

#### Methods

- `fit(dataset, labels=None, device='cpu')`: Fit the model and compute weights
- `get_difficulty_weights()`: Get raw difficulty weights
- `get_curriculum_weights(strategy='difficulty', current_iteration=0, warmup_iterations=None, min_temp=0.3, max_temp=2.0)`: Get curriculum weights with automatic temperature computation
- `save_weights(filepath)`: Save computed weights
- `load_weights(filepath)`: Load weights from file

### Automatic Temperature Computation

🆕 **New Feature**: Temperature is now computed automatically based on effective dataset size!

- **Early training** (small effective dataset): Higher temperature = more uniform sampling
- **Later training** (large effective dataset): Lower temperature = more focused sampling
- **Customizable range**: Use `min_temp` and `max_temp` parameters to control the temperature range
- **No manual tuning**: Users no longer need to manually set temperature values

### PretrainedModelExtractor

Feature extractor for PyTorch models.

#### Parameters

- `model`: PyTorch model
- `feature_layer`: Layer name for intermediate features (None for final layer)
- `normalize_features`: Whether to normalize features (default: True)

### create_curriculum_learning()

Convenience function for quick setup with custom models.

#### Parameters

- `model`: PyTorch model
- `dataset`: PyTorch dataset
- `n_clusters`: Number of clusters
- `use_labels`: Use supervised clustering
- `device`: Computing device
- `initial_effective_percentage`: Initial percentage of dataset to use (default: 0.3)
- `warmup_iterations`: Number of warmup iterations (default: 1000)
- `**kwargs`: Additional CurriculumLearning parameters

#### Returns

- `(difficulty_weights, curriculum_instance)`: Tuple of weights and curriculum object

### create_curriculum_learning_with_dino() (Recommended)

Convenience function specifically for DINO models.

#### Parameters

- `dataset`: PyTorch dataset
- `n_clusters`: Number of clusters (None for auto-selection)
- `use_labels`: Use supervised clustering
- `device`: Computing device
- `dino_model`: DINO variant ('dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8') 
- `initial_effective_percentage`: Initial percentage of dataset to use (default: 0.3)
- `warmup_iterations`: Number of warmup iterations (default: 1000)
- `**kwargs`: Additional CurriculumLearning parameters

#### Returns

- `(difficulty_weights, curriculum_instance)`: Tuple of weights and curriculum object

## Best Practices

### 1. Large-Scale Training Optimization
- **Batch Size**: Use larger batch sizes (256-512) for better GPU utilization and efficiency
- **Memory Management**: Enable `use_minibatch_kmeans` for datasets >10k samples
- **Initial Percentage**: Start with 10-20% for very large datasets (ImageNet-scale)
- **Warmup Iterations**: Use longer warmup (3000-8000) for stability at scale

### 2. Domain-Specific Recommendations
- **Image Generation**: Start with 15-25% of data, focus on prototypical examples
- **Audio Pretraining**: Use 10-15% initially, longer warmup for temporal patterns
- **Tokenizer Training**: Begin with 8-12%, emphasize common linguistic patterns
- **Language Models**: Use supervised clustering when possible

### 3. Model Selection (Recommended: DINO)
- **DINO models** (recommended): Excellent for curriculum learning with superior feature representations
  - `dino_vits16`: Good balance of performance and speed 
  - `dino_vitb16`: Better features but slower
  - `dino_vits8`: Faster but smaller patches

### 4. Curriculum Progression
- **Initial Effective Percentage**: 
  - Small datasets (<10k): 20-40%
  - Medium datasets (10k-100k): 15-25%  
  - Large datasets (>100k): 10-15%
- **Warmup Iterations**: Scale with dataset size and complexity
  - Small datasets: 1000-2000 iterations
  - Medium datasets: 2000-5000 iterations
  - Large datasets: 5000-10000 iterations

### 5. Automatic Temperature (New!)
- **No manual tuning required**: Temperature is automatically computed based on effective dataset size
- **Adaptive behavior**: Higher temperature for small effective datasets (exploration), lower for large (exploitation)
- **Customizable range**: Use `min_temp=0.3` to `max_temp=2.0` (defaults work well for most cases)
- **Optimal progression**: Automatically transitions from uniform sampling to focused sampling

## Performance Tips

1. **GPU Usage**: Always use GPU for feature extraction when available
2. **Batch Processing**: Larger batch sizes are more efficient for feature extraction and training
3. **Feature Caching**: Save computed weights to avoid recomputation
4. **Memory Optimization**: Use MiniBatchKMeans for datasets >10k samples
5. **Progressive Training**: Let automatic temperature handle the curriculum progression
6. **Large-Scale Datasets**: Use ImageNet or similar scale datasets for realistic evaluation
7. **Multi-GPU Training**: Scale batch sizes appropriately for multi-GPU setups

## Citation

If you use this library in your research, please cite our CVPR 2025 paper:

```bibtex
@inproceedings{lin2025prototypes,
  title={From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling},
  author={Lin, Jinhong and Wu, Cheng-En and Li, Huanran and Zhang, Jifan and Hu, Yu Hen and Morgado, Pedro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  url={https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf}
}
```

**Paper Link**: [From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_From_Prototypes_to_General_Distributions_An_Efficient_Curriculum_for_Masked_CVPR_2025_paper.pdf)

