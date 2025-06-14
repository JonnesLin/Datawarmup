# Curriculum Learning Library

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

### Basic Usage with DINO (Recommended)

```python
import torch
import torchvision
from curriculum_learning import create_curriculum_learning_with_dino

# Load your dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms)

# Generate curriculum weights using DINO (recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
difficulty_weights, curriculum = create_curriculum_learning_with_dino(
    dataset=dataset,
    device=device,
    dino_model='dino_vits16',  # Default DINO variant
    initial_effective_percentage=0.3,  # Start with 30% of dataset
    warmup_iterations=1000  # Warmup for 1000 iterations
)

# Use weights for training with curriculum progression
from torch.utils.data import WeightedRandomSampler
curriculum_weights = curriculum.get_curriculum_weights(
    temperature=0.5, 
    strategy='difficulty',
    current_iteration=current_iter,
    warmup_iterations=1000
)
sampler = WeightedRandomSampler(curriculum_weights, len(dataset), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Alternative: Custom Model

```python
import torch
import torchvision
from curriculum_learning import create_curriculum_learning

# Load your pretrained model
model = torchvision.models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier

# Load your dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms)

# Generate curriculum weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
difficulty_weights, curriculum = create_curriculum_learning(
    model=model,
    dataset=dataset,
    device=device,
    initial_effective_percentage=0.25,  # Start with 25% of dataset
    warmup_iterations=500
)

# Use weights for training
from torch.utils.data import WeightedRandomSampler
curriculum_weights = curriculum.get_curriculum_weights(
    temperature=0.5, 
    strategy='difficulty',
    current_iteration=current_iter,
    warmup_iterations=500
)
sampler = WeightedRandomSampler(curriculum_weights, len(dataset), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Advanced Usage

```python
from curriculum_learning import CurriculumLearning, PretrainedModelExtractor
import timm

# Create DINO model
model = timm.create_model('dino_vits16', pretrained=True, num_classes=0)

# Create custom feature extractor
extractor = PretrainedModelExtractor(
    model=model,
    feature_layer=None,  # Use final features
    normalize_features=True
)

# Configure curriculum learning with large-scale optimizations
curriculum = CurriculumLearning(
    feature_extractor=extractor,
    n_clusters=None,  # Auto-select optimal number
    auto_select_clusters=True,
    cluster_range=(5, 30),
    use_labels=True,  # Supervised clustering
    normalize_weights=True,
    batch_size=512,  # Large batch for efficiency
    initial_effective_percentage=0.2,  # Start with 20% of dataset
    warmup_iterations=2000  # Longer warmup for large datasets
)

# Fit and extract weights
curriculum.fit(dataset, device=device)
weights = curriculum.get_difficulty_weights()

# Save weights for later use
curriculum.save_weights('my_curriculum_weights.npy')
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
- `get_curriculum_weights(temperature=1.0, strategy='difficulty', current_iteration=0, warmup_iterations=None)`: Get curriculum weights for sampling
- `save_weights(filepath)`: Save computed weights
- `load_weights(filepath)`: Load weights from file

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

## Examples

### Example 1: Large-Scale Training with DINO

```python
from curriculum_learning import create_curriculum_learning_with_dino

# Optimize for large-scale training
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=large_dataset,  # Millions of samples
    n_clusters=None,  # Auto-select optimal number
    use_labels=False,  # Unsupervised clustering
    device=device,
    dino_model='dino_vits16',
    batch_size=1024,  # Large batch for efficiency
    initial_effective_percentage=0.1,  # Start with 10% for very large datasets
    warmup_iterations=5000,  # Longer warmup for stability
    use_minibatch_kmeans=True  # Essential for large datasets
)

print(f"Optimal clusters: {curriculum.optimal_clusters_}")
print(f"Training acceleration: Focus on {curriculum.initial_effective_percentage*100}% of data initially")
```

### Example 2: Audio Pretraining Application

```python
from curriculum_learning import create_curriculum_learning_with_dino

# Apply to audio pretraining (convert audio to spectrograms)
audio_dataset = AudioSpectrogramDataset(root='./audio_data')

weights, curriculum = create_curriculum_learning_with_dino(
    dataset=audio_dataset,
    device=device,
    dino_model='dino_vits16',
    initial_effective_percentage=0.2,  # Start with easier audio samples
    warmup_iterations=2000,
    n_clusters=15
)

# Use for audio model training
print("Curriculum learning applied to audio pretraining!")
```

### Example 3: Progressive Training with Warmup

```python
# Training loop with curriculum progression
for epoch in range(num_epochs):
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i
        
        # Get curriculum weights for current iteration
        curriculum_weights = curriculum.get_curriculum_weights(
            temperature=0.5,
            strategy='difficulty',
            current_iteration=current_iter,
            warmup_iterations=1000
        )
        
        # Update sampler with new weights
        sampler = WeightedRandomSampler(curriculum_weights, len(dataset), replacement=True)
        
        # Your training code here
        train_step(batch_data, batch_labels)
```

### Example 4: Multi-Domain Research Application

```python
# Example for tokenizer training
tokenizer_dataset = TokenizerTrainingDataset('./text_data')

weights, curriculum = create_curriculum_learning_with_dino(
    dataset=tokenizer_dataset,
    device=device,
    initial_effective_percentage=0.15,  # Start with simpler text patterns
    warmup_iterations=3000
)

print("Curriculum learning for tokenizer training - focus on common patterns first!")
```

## Best Practices

### 1. Large-Scale Training Optimization
- **Batch Size**: Use larger batch sizes (512-1024) for better GPU utilization
- **Memory Management**: Enable `use_minibatch_kmeans` for datasets >10k samples
- **Initial Percentage**: Start with 10-30% for very large datasets
- **Warmup Iterations**: Use longer warmup (2000-5000) for stability

### 2. Domain-Specific Recommendations
- **Image Generation**: Start with 20-30% of data, focus on prototypical examples
- **Audio Pretraining**: Use 15-25% initially, longer warmup for temporal patterns
- **Tokenizer Training**: Begin with 10-20%, emphasize common linguistic patterns
- **Language Models**: Use supervised clustering when possible

### 3. Model Selection (Recommended: DINO)
- **DINO models** (recommended): Excellent for curriculum learning with superior feature representations
  - `dino_vits16`: Good balance of performance and speed 
  - `dino_vitb16`: Better features but slower
  - `dino_vits8`: Faster but smaller patches

### 4. Curriculum Progression
- **Initial Effective Percentage**: 
  - Small datasets (<10k): 30-50%
  - Medium datasets (10k-100k): 20-30%  
  - Large datasets (>100k): 10-20%
- **Warmup Iterations**: Scale with dataset size and complexity

### 5. Temperature Tuning
- Lower temperature (0.1-0.5): More focused on difficult/easy samples
- Higher temperature (1.0-2.0): More uniform sampling
- Start with 0.5 and adjust based on training performance

## Performance Tips

1. **GPU Usage**: Always use GPU for feature extraction when available
2. **Batch Processing**: Larger batch sizes are more efficient for feature extraction
3. **Feature Caching**: Save computed weights to avoid recomputation
4. **Memory Optimization**: Use MiniBatchKMeans for datasets >10k samples
5. **Progressive Training**: Gradually increase effective dataset percentage during training

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

## License

MIT License - see LICENSE file for details. 
