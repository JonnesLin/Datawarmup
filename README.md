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
pip install -r requirements.txt
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

## License

MIT License - see LICENSE file for details.

## 🌡️ New Temperature Mechanism (v2.0)

The curriculum learning library now features an advanced temperature control mechanism that automatically determines optimal temperatures based on effective dataset size using **binary search** and provides **uniform distribution** after curriculum learning completion.

### Key Changes

#### 🔍 Binary Search Temperature Control
- **No more manual temperature parameters**: `min_temp` and `max_temp` parameters have been removed from user-facing methods
- **Automatic temperature computation**: Temperature is now computed automatically based on the target effective dataset size using binary search
- **Entropy-based effective size**: Uses entropy-based measures to determine actual effective dataset size
- **Theoretical maximum**: Automatically computes maximum temperature for theoretical maximum effective dataset size (100%)

#### 🎯 Uniform Distribution Post-Curriculum
- **Automatic transition**: After curriculum learning warmup iterations complete, the system automatically switches to uniform distribution
- **Equal sample weights**: All samples receive equal weight (multiply by 0 + 1 = uniform weighting)
- **No temperature needed**: Temperature computation is not applicable in the uniform phase

### Updated API

```python
# Before (v1.x) - Manual temperature parameters
curriculum_weights = curriculum.get_curriculum_weights(
    strategy='difficulty',
    current_iteration=iteration,
    min_temp=0.3,  # ❌ Removed
    max_temp=2.0   # ❌ Removed
)

# Now (v2.0) - Automatic temperature with binary search
curriculum_weights = curriculum.get_curriculum_weights(
    strategy='difficulty',
    current_iteration=iteration  # ✅ Temperature computed automatically
)
```

### Temperature Computation Process

1. **During Curriculum Learning** (iteration < warmup_iterations):
   - Calculate target effective dataset percentage based on progression
   - Use binary search to find temperature that achieves target effective size
   - Apply curriculum weighting based on difficulty/easy strategy

2. **After Curriculum Learning** (iteration >= warmup_iterations):
   - Return uniform distribution (all samples have equal weight)
   - No temperature computation needed

### Example Usage

```python
# Setup curriculum learning
weights, curriculum = create_curriculum_learning_with_dino(
    dataset=dataset,
    initial_effective_percentage=0.3,
    warmup_iterations=1000
)

# During training progression
for iteration in [0, 500, 1000, 1500]:  # 1000 = warmup_iterations
    curriculum_weights = curriculum.get_curriculum_weights(
        strategy='difficulty',
        current_iteration=iteration
    )
    
    if iteration >= curriculum.warmup_iterations:
        print(f"Iteration {iteration}: UNIFORM DISTRIBUTION")
        # All samples have equal weight: 1/n
    else:
        print(f"Iteration {iteration}: CURRICULUM LEARNING") 
        # Automatic temperature via binary search
```

### Advanced Features

```python
# Manual temperature computation for analysis
effective_size = 0.7  # 70% effective dataset
temperature = curriculum._compute_temperature(effective_size, strategy='difficulty')

# Get maximum temperature for theoretical maximum effective size
max_temp = curriculum._compute_max_temperature(strategy='difficulty')

# Compute effective dataset size for given temperature (for analysis)
effective_size = curriculum._compute_effective_dataset_size(
    weights=curriculum.weights_, 
    temperature=1.5, 
    strategy='difficulty'
)
``` 
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
