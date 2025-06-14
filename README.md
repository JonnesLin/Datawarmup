#!/usr/bin/env python3
"""
Example usage of the Curriculum Learning library

This script demonstrates how to use the curriculum learning library with:
1. Vision models (ResNet, ViT, etc.)
2. Different datasets (CIFAR-10, ImageNet, custom datasets)
3. Both supervised and unsupervised clustering approaches
4. Auto-selection of optimal cluster numbers
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
    create_curriculum_learning
)


def example_with_resnet_cifar10():
    """Example using ResNet50 with CIFAR-10 dataset"""
    print("=== Example 1: ResNet50 + CIFAR-10 ===")
    
    # Load pretrained ResNet50
    model = torchvision.models.resnet50(pretrained=True)
    # Remove final classification layer to get features
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use the convenience function for quick setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    weights, curriculum = create_curriculum_learning(
        model=model,
        dataset=dataset,
        n_clusters=None,  # Auto-select optimal number
        use_labels=False,  # Unsupervised clustering
        device=device,
        batch_size=128,
        cluster_range=(5, 25)
    )
    
    print(f"Difficulty weights shape: {weights.shape}")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # Get curriculum weights for training (focusing on difficult samples)
    curriculum_weights = curriculum.get_curriculum_weights(
        temperature=0.5, 
        strategy='difficulty'
    )
    
    # Create weighted sampler for training
    sampler = WeightedRandomSampler(
        weights=curriculum_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    print(f"Created curriculum-based data loader with {len(train_loader)} batches")
    
    return weights, curriculum


def example_with_vit_supervised():
    """Example using Vision Transformer with supervised clustering"""
    print("\n=== Example 2: Vision Transformer + Supervised Clustering ===")
    
    # Load pretrained Vision Transformer
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    model.eval()
    
    # Load dataset (using CIFAR-10 again for simplicity)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create feature extractor
    extractor = PretrainedModelExtractor(model, normalize_features=True)
    
    # Create curriculum learning with supervised clustering
    curriculum = CurriculumLearning(
        feature_extractor=extractor,
        n_clusters=5,  # Fixed number of clusters per class
        use_labels=True,  # Use supervised clustering
        normalize_weights=True,
        auto_select_clusters=False,
        batch_size=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum.fit(dataset, device=device)
    
    weights = curriculum.get_difficulty_weights()
    print(f"Supervised difficulty weights shape: {weights.shape}")
    
    # Analyze weights per class
    labels = np.array(dataset.targets)
    for class_idx in range(10):
        class_mask = labels == class_idx
        class_weights = weights[class_mask]
        print(f"Class {class_idx}: mean difficulty = {class_weights.mean():.4f}")
    
    return weights, curriculum


def example_with_custom_feature_layer():
    """Example extracting features from intermediate layers"""
    print("\n=== Example 3: Custom Feature Layer Extraction ===")
    
    # Load ResNet and extract features from layer3
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # Create extractor that gets features from 'layer3' instead of final layer
    extractor = PretrainedModelExtractor(
        model=model,
        feature_layer='layer3',  # Extract from intermediate layer
        normalize_features=True
    )
    
    # Simple dataset for demonstration
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use a smaller subset for faster processing
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use only first 1000 samples for demonstration
    subset_indices = list(range(1000))
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    curriculum = CurriculumLearning(
        feature_extractor=extractor,
        n_clusters=None,  # Auto-select
        auto_select_clusters=True,
        cluster_range=(3, 15),
        use_labels=False,
        batch_size=32
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum.fit(dataset, device=device)
    
    weights = curriculum.get_difficulty_weights()
    print(f"Custom layer weights shape: {weights.shape}")
    print(f"Optimal clusters found: {curriculum.optimal_clusters_}")
    
    return weights, curriculum


def example_save_load_weights():
    """Example of saving and loading curriculum weights"""
    print("\n=== Example 4: Save/Load Weights ===")
    
    # Create a simple curriculum learning setup
    model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use small subset for speed
    subset_indices = list(range(500))
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    weights, curriculum = create_curriculum_learning(
        model=model,
        dataset=dataset,
        n_clusters=8,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=32
    )
    
    # Save weights
    curriculum.save_weights('curriculum_weights.npy')
    print("Weights saved to 'curriculum_weights.npy'")
    
    # Create new curriculum instance and load weights
    extractor = PretrainedModelExtractor(model)
    new_curriculum = CurriculumLearning(extractor)
    new_curriculum.load_weights('curriculum_weights.npy')
    
    loaded_weights = new_curriculum.get_difficulty_weights()
    print(f"Loaded weights shape: {loaded_weights.shape}")
    print(f"Weights match: {np.allclose(weights, loaded_weights)}")
    
    return weights, curriculum


def example_temperature_strategies():
    """Example showing different temperature and strategy settings"""
    print("\n=== Example 5: Temperature and Strategy Comparison ===")
    
    # Quick setup with small dataset
    model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    subset_indices = list(range(1000))
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    weights, curriculum = create_curriculum_learning(
        model=model,
        dataset=dataset,
        n_clusters=10,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=64
    )
    
    # Test different temperature and strategy combinations
    strategies = ['difficulty', 'easy']
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    print("\nStrategy and Temperature Effects:")
    print("Strategy\tTemp\tMax Weight\tMin Weight\tStd")
    print("-" * 50)
    
    for strategy in strategies:
        for temp in temperatures:
            curriculum_weights = curriculum.get_curriculum_weights(
                temperature=temp, 
                strategy=strategy
            )
            print(f"{strategy}\t\t{temp}\t{curriculum_weights.max():.4f}\t\t"
                  f"{curriculum_weights.min():.4f}\t\t{curriculum_weights.std():.4f}")
    
    return weights, curriculum


def main():
    """Run all examples"""
    print("Curriculum Learning Library Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic usage with ResNet
        example_with_resnet_cifar10()
        
        # Example 2: Supervised clustering with ViT
        example_with_vit_supervised()
        
        # Example 3: Custom feature layer
        example_with_custom_feature_layer()
        
        # Example 4: Save/Load functionality
        example_save_load_weights()
        
        # Example 5: Temperature and strategy effects
        example_temperature_strategies()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
