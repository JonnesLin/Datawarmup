# Curriculum Learning Library

A ready-to-use curriculum learning library that uses K-means clustering on pretrained model features to generate difficulty-based sample weights for improved training.

## Overview

This library implements a curriculum learning approach where:
1. **Feature Extraction**: Features are extracted from your data using any pretrained model
2. **Clustering**: K-means clustering groups similar samples together
3. **Difficulty Assessment**: Samples farther from cluster centers are considered more difficult
4. **Curriculum Weights**: Generated weights can be used for weighted sampling during training

## Key Features

- ✅ **Universal Compatibility**: Works with any PyTorch model and dataset
- ✅ **Auto-Cluster Selection**: Automatically finds optimal number of clusters using Davies-Bouldin index
- ✅ **Supervised & Unsupervised**: Supports both clustering approaches
- ✅ **Memory Efficient**: Handles large datasets with batched processing
- ✅ **Intermediate Features**: Extract features from any model layer
- ✅ **Save/Load Weights**: Persist computed weights for reuse

## Installation

```bash
pip install -r curriculum_requirements.txt
```

## Quick Start

### Basic Usage

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
    device=device
)

# Use weights for training
from torch.utils.data import WeightedRandomSampler
curriculum_weights = curriculum.get_curriculum_weights(temperature=0.5, strategy='difficulty')
sampler = WeightedRandomSampler(curriculum_weights, len(dataset), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Advanced Usage

```python
from curriculum_learning import CurriculumLearning, PretrainedModelExtractor

# Create custom feature extractor
extractor = PretrainedModelExtractor(
    model=your_model,
    feature_layer='layer3',  # Extract from intermediate layer
    normalize_features=True
)

# Configure curriculum learning
curriculum = CurriculumLearning(
    feature_extractor=extractor,
    n_clusters=None,  # Auto-select optimal number
    auto_select_clusters=True,
    cluster_range=(5, 30),
    use_labels=True,  # Supervised clustering
    normalize_weights=True,
    batch_size=128
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
- `random_state`: Random seed (default: 42)

#### Methods

- `fit(dataset, labels=None, device='cpu')`: Fit the model and compute weights
- `get_difficulty_weights()`: Get raw difficulty weights
- `get_curriculum_weights(temperature=1.0, strategy='difficulty')`: Get curriculum weights for sampling
- `save_weights(filepath)`: Save computed weights
- `load_weights(filepath)`: Load weights from file

### PretrainedModelExtractor

Feature extractor for PyTorch models.

#### Parameters

- `model`: PyTorch model
- `feature_layer`: Layer name for intermediate features (None for final layer)
- `normalize_features`: Whether to normalize features (default: True)

### create_curriculum_learning()

Convenience function for quick setup.

#### Parameters

- `model`: PyTorch model
- `dataset`: PyTorch dataset
- `n_clusters`: Number of clusters
- `use_labels`: Use supervised clustering
- `device`: Computing device
- `**kwargs`: Additional CurriculumLearning parameters

#### Returns

- `(difficulty_weights, curriculum_instance)`: Tuple of weights and curriculum object

## Examples

### Example 1: Vision Transformer with Auto-Clustering

```python
import timm
from curriculum_learning import create_curriculum_learning

# Load Vision Transformer
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

# Auto-select optimal clusters
weights, curriculum = create_curriculum_learning(
    model=model,
    dataset=your_dataset,
    n_clusters=None,  # Auto-select
    cluster_range=(5, 25),
    device=device
)

print(f"Optimal clusters: {curriculum.optimal_clusters_}")
```

### Example 2: Supervised Clustering per Class

```python
from curriculum_learning import CurriculumLearning, PretrainedModelExtractor

extractor = PretrainedModelExtractor(model)
curriculum = CurriculumLearning(
    feature_extractor=extractor,
    n_clusters=5,  # 5 clusters per class
    use_labels=True,  # Cluster within each class
    auto_select_clusters=False
)

curriculum.fit(dataset, device=device)
weights = curriculum.get_difficulty_weights()

# Analyze per-class difficulty
labels = np.array(dataset.targets)
for class_idx in range(num_classes):
    class_mask = labels == class_idx
    class_weights = weights[class_mask]
    print(f"Class {class_idx} mean difficulty: {class_weights.mean():.4f}")
```

### Example 3: Custom Feature Layer

```python
# Extract features from ResNet's layer3 instead of final layer
extractor = PretrainedModelExtractor(
    model=resnet_model,
    feature_layer='layer3',
    normalize_features=True
)

curriculum = CurriculumLearning(extractor, n_clusters=10)
curriculum.fit(dataset, device=device)
```

### Example 4: Temperature and Strategy Effects

```python
# Get weights for different curriculum strategies
easy_weights = curriculum.get_curriculum_weights(
    temperature=0.5, 
    strategy='easy'  # Focus on easier samples first
)

hard_weights = curriculum.get_curriculum_weights(
    temperature=0.5, 
    strategy='difficulty'  # Focus on harder samples
)

# Lower temperature = more focused sampling
focused_weights = curriculum.get_curriculum_weights(temperature=0.1)
uniform_weights = curriculum.get_curriculum_weights(temperature=2.0)
```

## Best Practices

### 1. Model Selection
- Use pretrained models for better feature representations
- For images: ResNet, EfficientNet, Vision Transformers work well
- For text: BERT, RoBERTa, etc.
- For other domains: Any pretrained model in your domain

### 2. Cluster Selection
- Start with auto-selection to find optimal number
- For supervised clustering: 3-10 clusters per class typically work well
- For unsupervised: 10-50 clusters depending on dataset size

### 3. Temperature Tuning
- Lower temperature (0.1-0.5): More focused on difficult/easy samples
- Higher temperature (1.0-2.0): More uniform sampling
- Start with 0.5 and adjust based on training performance

### 4. Memory Management
- Use `batch_size` parameter to control memory usage
- Enable `use_minibatch_kmeans` for large datasets (>10k samples)
- Consider using intermediate layers for very large models

### 5. Integration with Training
- Update curriculum weights periodically during training
- Start with easier samples, gradually focus on harder ones
- Monitor training metrics to adjust temperature

## Performance Tips

1. **GPU Usage**: Always use GPU for feature extraction when available
2. **Batch Processing**: Larger batch sizes are more efficient for feature extraction
3. **Feature Caching**: Save computed weights to avoid recomputation
4. **Memory Optimization**: Use MiniBatchKMeans for datasets >10k samples

## Troubleshooting

**Q: Out of memory during feature extraction**
A: Reduce `batch_size` parameter or use a smaller model

**Q: Clustering takes too long**
A: Enable `use_minibatch_kmeans` or reduce `cluster_range`

**Q: Poor curriculum weights**
A: Try different `n_clusters` values or switch between supervised/unsupervised

**Q: Model outputs wrong format**
A: Ensure your model outputs feature vectors, not classifications

## Citation

If you use this library in your research, please cite:

```bibtex
@software{curriculum_learning_library,
  title={Curriculum Learning with K-means Clustering},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/curriculum-learning}
}
```

## License

MIT License - see LICENSE file for details. 
