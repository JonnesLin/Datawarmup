#!/usr/bin/env python3
"""
Integration example showing how to use curriculum learning in a real training loop

This example demonstrates:
1. Setting up curriculum learning with initial effective percentage and warmup
2. Integrating with PyTorch training loops
3. Dynamic curriculum progression during training
4. Multi-domain applications (vision, audio, text)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from curriculum_learning import create_curriculum_learning_with_dino


class SimpleClassifier(nn.Module):
    """Simple CNN classifier for demonstration"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_with_curriculum_progression(model, train_loader, test_loader, curriculum, 
                                    device, num_epochs=10, warmup_iterations=2000):
    """
    Training loop with curriculum progression (temperature now automatic)
    
    Args:
        model: PyTorch model to train
        train_loader: Initial training data loader
        test_loader: Test data loader for evaluation
        curriculum: CurriculumLearning instance
        device: Device to train on
        num_epochs: Number of training epochs
        warmup_iterations: Number of iterations for curriculum warmup
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    # Track training metrics
    iteration = 0
    epoch_losses = []
    effective_percentages = []
    
    print(f"Training with automatic curriculum progression over {num_epochs} epochs")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Initial effective percentage: {curriculum.initial_effective_percentage:.1%}")
    print("Temperature is now computed automatically based on effective dataset size")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Update curriculum weights for this epoch (temperature automatic)
        curriculum_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iteration
        )
        
        # Calculate effective dataset percentage
        active_samples = (curriculum_weights > 1e-5).sum()
        effective_percentage = active_samples / len(curriculum_weights)
        effective_percentages.append(effective_percentage)
        
        # Show automatic temperature computation
        auto_temp = curriculum._compute_temperature(effective_percentage)
        
        # Create new sampler with updated weights
        sampler = WeightedRandomSampler(
            weights=curriculum_weights,
            num_samples=len(train_loader.dataset),
            replacement=True
        )
        
        # Create new data loader with updated sampler
        current_loader = DataLoader(
            train_loader.dataset, 
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers
        )
        
        # Training loop for this epoch
        pbar = tqdm(current_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            iteration += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Effective': f'{effective_percentage:.1%}',
                'Auto Temp': f'{auto_temp:.3f}',
                'Iter': iteration
            })
        
        avg_epoch_loss = epoch_loss / epoch_batches
        epoch_losses.append(avg_epoch_loss)
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        
        print(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, "
              f"Test Acc={test_accuracy:.2%}, Effective Data={effective_percentage:.1%}, "
              f"Auto Temp={auto_temp:.3f}")
    
    return epoch_losses, effective_percentages


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    model.train()
    return correct / total


def vision_training_example():
    """Complete vision training example with curriculum learning on ImageNet"""
    print("=== Large-Scale Vision Training with Curriculum Learning (ImageNet) ===")
    
    # Setup dataset
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet
    # Note: ImageNet dataset requires manual download and setup
    train_dataset = torchvision.datasets.ImageNet(
        root='./data/imagenet', split='train', transform=transform_train
    )
    test_dataset = torchvision.datasets.ImageNet(
        root='./data/imagenet', split='val', transform=transform_test
    )
    
    # Use meaningful subset for large-scale demonstration
    train_indices = list(range(50000))  # Use 50k samples for realistic training
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    test_indices = list(range(5000))   # Use 5k test samples
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Setup curriculum learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Setting up large-scale curriculum learning...")
    weights, curriculum = create_curriculum_learning_with_dino(
        dataset=train_dataset,
        n_clusters=None,  # Auto-select
        use_labels=True,  # Supervised clustering
        device=device,
        dino_model='dino_vits16',
        batch_size=128,  # Larger batch for ImageNet
        initial_effective_percentage=0.15,  # Start with 15% for large-scale
        warmup_iterations=2000  # Appropriate warmup for 50k samples
    )
    
    print(f"Curriculum setup complete:")
    print(f"  - Optimal clusters: {curriculum.optimal_clusters_}")
    print(f"  - Initial effective percentage: {curriculum.initial_effective_percentage:.1%}")
    print(f"  - Warmup iterations: {curriculum.warmup_iterations}")
    
    # Create initial data loaders with larger batch sizes
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Create model (updated for ImageNet's 1000 classes)
    model = SimpleClassifier(num_classes=1000)
    
    # Train with curriculum
    losses, effective_percentages = train_with_curriculum_progression(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        curriculum=curriculum,
        device=device,
        num_epochs=3,  # Fewer epochs for large-scale demo
        warmup_iterations=2000
    )
    
    print(f"\nLarge-scale training completed!")
    print(f"Final effective dataset percentage: {effective_percentages[-1]:.1%}")
    
    return model, losses, effective_percentages


def audio_pretraining_example():
    """Example of curriculum learning for large-scale audio pretraining"""
    print("\n=== Large-Scale Audio Pretraining with Curriculum Learning ===")
    
    # Simulate large-scale audio spectrograms
    print("Creating large-scale simulated audio spectrogram dataset...")
    n_samples = 30000  # 30k samples for realistic audio pretraining demo
    n_classes = 15  # 15 audio categories
    
    # Generate synthetic spectrograms (in practice, load real audio data)
    spectrograms = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Add structure to make some samples "harder"
    for i in range(n_classes):
        class_mask = labels == i
        class_indices = torch.where(class_mask)[0]
        # Make 30% of each class harder by adding noise
        hard_count = len(class_indices) // 3
        hard_indices = class_indices[:hard_count]
        spectrograms[hard_indices] += torch.randn_like(spectrograms[hard_indices]) * 0.3
    
    audio_dataset = TensorDataset(spectrograms, labels)
    
    # Setup curriculum for large-scale audio
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Setting up large-scale audio curriculum learning...")
    weights, curriculum = create_curriculum_learning_with_dino(
        dataset=audio_dataset,
        n_clusters=20,  # More clusters for audio diversity at scale
        use_labels=False,  # Unsupervised clustering
        device=device,
        dino_model='dino_vits16',
        batch_size=64,  # Larger batch for audio efficiency
        initial_effective_percentage=0.12,  # Start with 12% for audio complexity
        warmup_iterations=2500  # Longer warmup for audio patterns
    )
    
    print(f"Audio curriculum setup:")
    print(f"  - Clusters: {curriculum.optimal_clusters_}")
    print(f"  - Initial effective: {curriculum.initial_effective_percentage:.1%}")
    print(f"  - Warmup iterations: {curriculum.warmup_iterations}")
    
    # Demonstrate progression with automatic temperature
    print("\nAudio curriculum progression (automatic temperature):")
    test_iterations = [0, 800, 1600, 2500, 3500]
    for iter_num in test_iterations:
        curr_weights = curriculum.get_curriculum_weights(
            strategy='difficulty', 
            current_iteration=iter_num
        )
        active = (curr_weights > 1e-5).sum()
        percentage = active / len(curr_weights)
        auto_temp = curriculum._compute_temperature(percentage)
        print(f"  Iteration {iter_num:4d}: {percentage:.1%} active ({active} samples), Auto temp = {auto_temp:.3f}")
    
    return curriculum


def tokenizer_training_example():
    """Example of curriculum learning for large-scale tokenizer training"""
    print("\n=== Large-Scale Tokenizer Training with Curriculum Learning ===")
    
    # Simulate large-scale text embeddings for tokenizer training
    print("Creating large-scale simulated text embedding dataset...")
    n_samples = 40000  # 40k samples for realistic tokenizer training demo
    embedding_dim = 768
    n_languages = 12
    
    # Generate synthetic text embeddings
    embeddings = torch.randn(n_samples, embedding_dim)
    language_labels = torch.randint(0, n_languages, (n_samples,))
    
    # Make some languages/domains harder by adding complexity
    for lang_id in range(n_languages):
        lang_mask = language_labels == lang_id
        lang_indices = torch.where(lang_mask)[0]
        
        # Make certain languages more complex
        if lang_id in [2, 5, 8, 10]:  # "Complex" languages
            embeddings[lang_indices] += torch.randn_like(embeddings[lang_indices]) * 0.4
    
    text_dataset = TensorDataset(embeddings, language_labels)
    
    # Create a simple text model for curriculum learning
    text_model = nn.Sequential(
        nn.Linear(embedding_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Setting up large-scale tokenizer curriculum learning...")
    from curriculum_learning import create_curriculum_learning
    
    weights, curriculum = create_curriculum_learning(
        model=text_model,
        dataset=text_dataset,
        n_clusters=None,  # Auto-select
        use_labels=True,  # Supervised by language
        device=device,
        batch_size=128,  # Larger batch for text efficiency
        initial_effective_percentage=0.08,  # Start small for tokenizer complexity
        warmup_iterations=4000  # Long warmup for language patterns
    )
    
    print(f"Tokenizer curriculum setup:")
    print(f"  - Optimal clusters: {curriculum.optimal_clusters_}")
    print(f"  - Initial effective: {curriculum.initial_effective_percentage:.1%}")
    print(f"  - Warmup iterations: {curriculum.warmup_iterations}")
    
    # Show progression specific to large-scale tokenizer training
    print("\nTokenizer training progression (automatic temperature):")
    test_iterations = [0, 1500, 3000, 4000, 6000]
    for iter_num in test_iterations:
        curr_weights = curriculum.get_curriculum_weights(
            strategy='difficulty',
            current_iteration=iter_num
        )
        active = (curr_weights > 1e-5).sum()
        percentage = active / len(curr_weights)
        phase = "Warmup" if iter_num < 4000 else "Post-warmup"
        auto_temp = curriculum._compute_temperature(percentage)
        print(f"  Iteration {iter_num:4d} ({phase:10s}): {percentage:.1%} active, Auto temp = {auto_temp:.3f}")
    
    return curriculum


def main():
    """Run integration examples"""
    print("Curriculum Learning - Training Integration Examples")
    print("=" * 60)
    
    try:
        # Example 1: Complete vision training pipeline
        model, losses, percentages = vision_training_example()
        
        # Example 2: Audio pretraining setup
        audio_curriculum = audio_pretraining_example()
        
        # Example 3: Tokenizer training setup
        tokenizer_curriculum = tokenizer_training_example()
        
        print("\n" + "=" * 60)
        print("🎉 All integration examples completed!")
        print("\n💡 Key Integration Features:")
        print("   ✓ Dynamic curriculum progression during training")
        print("   ✓ Automatic effective dataset percentage scaling")
        print("   ✓ Multi-domain application examples")
        print("   ✓ Real training loop integration")
        print("   ✓ Warmup iteration control")
        
        print("\n📊 Training Results Summary:")
        print(f"   - Vision training: {len(losses)} epochs completed")
        print(f"   - Final effective dataset: {percentages[-1]:.1%}")
        print(f"   - Audio curriculum: {audio_curriculum.warmup_iterations} warmup iterations")
        print(f"   - Tokenizer curriculum: {tokenizer_curriculum.warmup_iterations} warmup iterations")
        
    except Exception as e:
        print(f"Error in integration examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
