#!/usr/bin/env python3
"""
Example usage of the Curriculum Learning library

This script demonstrates how to use the curriculum learning library with:
1. DINO models (default and recommended)
2. Vision models (ResNet, ViT, etc.)
3. Different datasets (CIFAR-10, ImageNet, custom datasets)
4. Curriculum progression with initial effective percentage and warmup iterations
5. Both supervised and unsupervised clustering approaches
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import numpy as np
from typing import Tuple

from curriculum_learning import (
    CurriculumLearning, 
    PretrainedModelExtractor, 
    create_curriculum_learning,
    create_curriculum_learning_with_dino
)


def example_with_dino_imagenet():
    """Example using DINO (recommended default) with ImageNet dataset"""
    print("=== Example 1: DINO + ImageNet with Curriculum Progression ===")
    
    # Load ImageNet dataset with appropriate transforms for DINO
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # DINO expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet - using training set
    # Note: ImageNet dataset requires manual download and setup
    dataset = torchvision.datasets.ImageNet(
        root='./data/imagenet', split='train', transform=transform
    )
    
    # Use a meaningful subset for large-scale demonstration (100k samples)
    subset_indices = list(range(100000))  # 100k samples for realistic large-scale training
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # Use the convenience function with DINO as default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    weights, curriculum = create_curriculum_learning_with_dino(
        dataset=dataset,
        n_clusters=None,  # Auto-select optimal number
        use_labels=False,  # Unsupervised clustering for large-scale
        device=device,
        dino_model='dino_vits16',  # Default DINO variant
        batch_size=256,  # Large batch for efficiency
        cluster_range=(10, 50),  # More clusters for ImageNet diversity
        initial_effective_percentage=0.15,  # Start with 15% for large-scale
        warmup_iterations=5000  # Longer warmup for large-scale training
    )
    
    print(f"Difficulty weights shape: {weights.shape}")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Optimal clusters found: {curriculum.optimal_clusters_}")
    print(f"Initial effective percentage: {curriculum.initial_effective_percentage}")
    print(f"Warmup iterations: {curriculum.warmup_iterations}")
    
    # Simulate training progression (temperature is now automatic)
    print("\n--- Large-Scale Training Progression Simulation ---")
    for iteration in [0, 1000, 2500, 5000, 7500, 10000]:
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iteration
        )
        
        active_samples = (curriculum_weights > 1e-5).sum()
        effective_percentage = active_samples / len(curriculum_weights)
        
        # The temperature is now computed automatically based on effective percentage
        temp = curriculum._compute_temperature(effective_percentage)
        
        print(f"Iteration {iteration:5d}: Effective dataset = {effective_percentage:.2%} "
              f"({active_samples:6d}/{len(curriculum_weights)} samples), Auto temp = {temp:.3f}")
    
    return weights, curriculum


def example_audio_pretraining_simulation():
    """Example simulating audio pretraining scenario with large-scale data"""
    print("\n=== Example 2: Large-Scale Audio Pretraining Simulation ===")
    
    # Simulate large-scale audio spectrograms as 2D images
    # In practice, you would load actual audio spectrograms from a large dataset
    print("Simulating large-scale audio spectrogram dataset...")
    
    # Create dummy audio-like data (spectrograms) - larger scale
    n_samples = 50000  # 50k samples for realistic audio pretraining
    spectrogram_data = torch.randn(n_samples, 3, 224, 224)  # Simulate RGB spectrograms
    audio_labels = torch.randint(0, 20, (n_samples,))  # 20 audio classes (more realistic)
    
    from torch.utils.data import TensorDataset
    audio_dataset = TensorDataset(spectrogram_data, audio_labels)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Apply curriculum learning for large-scale audio pretraining
    weights, curriculum = create_curriculum_learning_with_dino(
        dataset=audio_dataset,
        n_clusters=25,  # More clusters for diverse audio patterns at scale
        use_labels=False,  # Unsupervised clustering
        device=device,
        dino_model='dino_vits16',
        batch_size=128,  # Larger batch for audio efficiency
        initial_effective_percentage=0.12,  # Start with 12% for audio complexity at scale
        warmup_iterations=4000  # Longer warmup for audio patterns
    )
    
    print(f"Audio curriculum weights shape: {weights.shape}")
    print(f"Weight statistics: min={weights.min():.4f}, max={weights.max():.4f}")
    print(f"Configuration: {curriculum.initial_effective_percentage:.1%} initial, "
          f"{curriculum.warmup_iterations} warmup iterations")
    
    # Demonstrate curriculum progression for large-scale audio (temperature automatic)
    print("\n--- Audio Training Progression ---")
    for iteration in [0, 1000, 2000, 4000, 6000]:
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iteration
        )
        
        active_samples = (curriculum_weights > 1e-5).sum()
        effective_percentage = active_samples / len(curriculum_weights)
        
        # Show automatic temperature computation
        temp = curriculum._compute_temperature(effective_percentage)
        
        print(f"Audio iteration {iteration:4d}: {effective_percentage:.2%} dataset active "
              f"({active_samples} samples), Auto temp = {temp:.3f}")
    
    return weights, curriculum


def example_tokenizer_training_simulation():
    """Example simulating large-scale tokenizer training scenario"""
    print("\n=== Example 3: Large-Scale Tokenizer Training Simulation ===")
    
    # Simulate large-scale text sequences as embeddings
    print("Simulating large-scale text sequence dataset for tokenizer training...")
    
    # Create dummy text embeddings - larger scale
    n_samples = 80000  # 80k samples for realistic tokenizer training
    sequence_length = 512
    embedding_dim = 768
    
    # Simulate text embeddings (in practice, these would be from a pretrained model)
    text_embeddings = torch.randn(n_samples, embedding_dim)
    text_labels = torch.randint(0, 50, (n_samples,))  # 50 language/domain labels
    
    from torch.utils.data import TensorDataset
    text_dataset = TensorDataset(text_embeddings, text_labels)
    
    # Create a simple embedding model for feature extraction
    embedding_model = nn.Sequential(
        nn.Linear(embedding_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Apply curriculum learning for large-scale tokenizer training
    weights, curriculum = create_curriculum_learning(
        model=embedding_model,
        dataset=text_dataset,
        n_clusters=None,  # Auto-select
        use_labels=True,  # Supervised clustering by language/domain
        device=device,
        batch_size=256,  # Large batch for text efficiency
        initial_effective_percentage=0.08,  # Start with 8% for tokenizer complexity at scale
        warmup_iterations=8000  # Very long warmup for language patterns
    )
    
    print(f"Tokenizer curriculum weights shape: {weights.shape}")
    print(f"Optimal clusters selected: {curriculum.optimal_clusters_}")
    print(f"Configuration: {curriculum.initial_effective_percentage:.1%} initial, "
          f"{curriculum.warmup_iterations} warmup iterations")
    
    # Show progression for large-scale tokenizer training (temperature automatic)
    print("\n--- Tokenizer Training Progression ---")
    for iteration in [0, 2000, 4000, 8000, 12000]:
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iteration
        )
        
        active_samples = (curriculum_weights > 1e-5).sum()
        effective_percentage = active_samples / len(curriculum_weights)
        
        # Show automatic temperature computation
        temp = curriculum._compute_temperature(effective_percentage)
        
        print(f"Tokenizer iteration {iteration:5d}: {effective_percentage:.2%} dataset active, Auto temp = {temp:.3f}")
    
    return weights, curriculum


def example_image_generation_simulation():
    """Example simulating large-scale image generation training"""
    print("\n=== Example 4: Large-Scale Image Generation Training Simulation ===")
    
    # Load a meaningful subset of ImageNet for image generation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use ImageNet for realistic image generation scenario
    dataset = torchvision.datasets.ImageNet(
        root='./data/imagenet', split='train', transform=transform
    )
    
    # Use meaningful subset for image generation (75k samples)
    subset_indices = list(range(75000))
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure for large-scale image generation (focus on prototypical examples first)
    weights, curriculum = create_curriculum_learning_with_dino(
        dataset=dataset,
        n_clusters=30,  # Good number for ImageNet diversity
        use_labels=True,  # Use class information for better clustering
        device=device,
        dino_model='dino_vitb16',  # Stronger model for image generation
        batch_size=128,  # Large batch for generation efficiency
        initial_effective_percentage=0.2,  # Start with 20% prototypical images
        warmup_iterations=3000  # Moderate warmup for image generation
    )
    
    print(f"Image generation curriculum weights shape: {weights.shape}")
    print(f"Weight statistics: min={weights.min():.4f}, max={weights.max():.4f}")
    print(f"Configuration: {curriculum.initial_effective_percentage:.1%} initial, "
          f"{curriculum.warmup_iterations} warmup iterations")
    
    # Demonstrate training phases (temperature now automatic)
    print("\n--- Image Generation Training Phases ---")
    phases = [
        (0, "Easy prototypes"),
        (1000, "Gradual expansion"),
        (2000, "Mid-warmup"),
        (3000, "Full warmup"),
        (4500, "Post-warmup")
    ]
    
    for iteration, phase_name in phases:
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iteration
        )
        
        active_samples = (curriculum_weights > 1e-5).sum()
        effective_percentage = active_samples / len(curriculum_weights)
        
        # Show automatic temperature computation
        temp = curriculum._compute_temperature(effective_percentage)
        
        print(f"{phase_name:15s} (iter {iteration:4d}): {effective_percentage:.2%} dataset, Auto temp = {temp:.3f}")
    
    return weights, curriculum


def main():
    """Run key examples demonstrating the new features"""
    print("Curriculum Learning Library - New Features Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic DINO usage with progression
        example_with_dino_imagenet()
        
        # Example 2: Audio pretraining simulation
        example_audio_pretraining_simulation()
        
        # Example 3: Tokenizer training simulation  
        example_tokenizer_training_simulation()
        
        # Example 4: Image generation simulation
        example_image_generation_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("💡 Key features demonstrated:")
        print("   - Initial effective percentage control")
        print("   - Warmup iteration progression") 
        print("   - Multi-domain applications")
        print("   - DINO integration (recommended)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
