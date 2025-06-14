import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances, davies_bouldin_score
from tqdm import tqdm
from typing import Optional, Union, Callable, Tuple, Any
import warnings
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract_features(self, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
        """Extract features from data using the model"""
        pass


class PretrainedModelExtractor(FeatureExtractor):
    """Feature extractor for pretrained PyTorch models"""
    
    def __init__(self, model: nn.Module, feature_layer: Optional[str] = None, 
                 normalize_features: bool = True):
        """
        Args:
            model: Pretrained PyTorch model
            feature_layer: Name of layer to extract features from (None for final features)
            normalize_features: Whether to normalize extracted features
        """
        self.model = model.eval()
        self.feature_layer = feature_layer
        self.normalize_features = normalize_features
        self.features = []
        
        if feature_layer:
            self._register_hook()
    
    def _register_hook(self):
        """Register forward hook for intermediate layer feature extraction"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features.append(output.detach())
            elif isinstance(output, (list, tuple)):
                self.features.append(output[0].detach())
        
        # Find and register hook on specified layer
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook_fn)
                break
    
    @torch.no_grad()
    def extract_features(self, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
        """Extract features from data loader"""
        self.model.to(device)
        self.model.eval()
        
        all_features = []
        
        for batch in tqdm(data_loader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]  # Assume first element is input
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            
            if self.feature_layer:
                self.features = []
                with torch.no_grad():
                    _ = self.model(inputs)
                features = self.features[0]
            else:
                with torch.no_grad():
                    features = self.model(inputs)
            
            # Handle different output formats
            if isinstance(features, torch.Tensor):
                if features.dim() > 2:
                    # Global average pooling for spatial features
                    features = features.mean(dim=list(range(2, features.dim())))
            
            all_features.append(features.cpu())
        
        features_tensor = torch.cat(all_features, dim=0)
        
        if self.normalize_features:
            # Standardize features
            mean = features_tensor.mean(dim=0, keepdim=True)
            std = features_tensor.std(dim=0, keepdim=True)
            features_tensor = (features_tensor - mean) / (std + 1e-6)
            
            # L2 normalize
            features_tensor = F.normalize(features_tensor, p=2, dim=1)
        
        return features_tensor


class CurriculumLearning:
    """
    Curriculum Learning using K-means clustering on pretrained model features
    
    This class implements a curriculum learning approach where:
    1. Features are extracted from data using a pretrained model
    2. K-means clustering is applied to group similar samples
    3. Difficulty weights are computed based on distance to cluster centers
    4. Samples farther from cluster centers are considered more difficult
    """
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor,
                 n_clusters: Optional[int] = None,
                 auto_select_clusters: bool = True,
                 cluster_range: Tuple[int, int] = (2, 20),
                 use_labels: bool = False,
                 normalize_weights: bool = True,
                 batch_size: int = 256,
                 use_minibatch_kmeans: bool = True,
                 initial_effective_percentage: float = 0.3,
                 warmup_iterations: int = 1000,
                 random_state: int = 42):
        """
        Args:
            feature_extractor: FeatureExtractor instance for extracting features
            n_clusters: Number of clusters (if None, will be auto-selected)
            auto_select_clusters: Whether to automatically select optimal number of clusters
            cluster_range: Range of cluster numbers to try for auto-selection
            use_labels: Whether to perform clustering within each class separately
            normalize_weights: Whether to normalize difficulty weights
            batch_size: Batch size for feature extraction
            use_minibatch_kmeans: Whether to use MiniBatchKMeans for large datasets
            initial_effective_percentage: Initial percentage of dataset to use (0.1 to 1.0)
            warmup_iterations: Number of warmup iterations for curriculum progression
            random_state: Random seed for reproducibility
        """
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        self.auto_select_clusters = auto_select_clusters
        self.cluster_range = cluster_range
        self.use_labels = use_labels
        self.normalize_weights = normalize_weights
        self.batch_size = batch_size
        self.use_minibatch_kmeans = use_minibatch_kmeans
        self.initial_effective_percentage = initial_effective_percentage
        self.warmup_iterations = warmup_iterations
        self.random_state = random_state
        
        # Validate parameters
        if not 0.1 <= initial_effective_percentage <= 1.0:
            raise ValueError("initial_effective_percentage must be between 0.1 and 1.0")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        
        # Internal state
        self.features_ = None
        self.labels_ = None
        self.weights_ = None
        self.optimal_clusters_ = None
    
    def _select_optimal_clusters(self, features: np.ndarray, 
                               labels: Optional[np.ndarray] = None) -> int:
        """
        Automatically select optimal number of clusters using Davies-Bouldin index
        Lower Davies-Bouldin index indicates better clustering
        """
        if labels is not None and self.use_labels:
            # For supervised case, use number of unique classes as upper bound
            max_clusters = min(len(np.unique(labels)), self.cluster_range[1])
            cluster_range = range(self.cluster_range[0], max_clusters + 1)
        else:
            cluster_range = range(self.cluster_range[0], self.cluster_range[1] + 1)
        
        best_score = float('inf')
        best_clusters = self.cluster_range[0]
        
        print("Selecting optimal number of clusters...")
        for n_clusters in tqdm(cluster_range, desc="Testing cluster numbers"):
            try:
                if self.use_minibatch_kmeans and len(features) > 10000:
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                           random_state=self.random_state,
                                           batch_size=min(1024, len(features)//10))
                else:
                    kmeans = KMeans(n_clusters=n_clusters, 
                                  random_state=self.random_state,
                                  n_init=10)
                
                cluster_labels = kmeans.fit_predict(features)
                
                # Calculate Davies-Bouldin index
                db_score = davies_bouldin_score(features, cluster_labels)
                
                if db_score < best_score:
                    best_score = db_score
                    best_clusters = n_clusters
                    
            except Exception as e:
                warnings.warn(f"Failed to fit {n_clusters} clusters: {e}")
                continue
        
        print(f"Optimal number of clusters: {best_clusters} (DB score: {best_score:.4f})")
        return best_clusters
    
    def _compute_difficulty_weights_supervised(self, features: np.ndarray, 
                                             labels: np.ndarray, 
                                             n_clusters: int) -> np.ndarray:
        """Compute difficulty weights using supervised clustering (within each class)"""
        weights = np.ones(len(features))
        
        for class_label in tqdm(np.unique(labels), desc="Processing classes"):
            class_mask = labels == class_label
            class_features = features[class_mask]
            
            if len(class_features) < n_clusters:
                # If fewer samples than clusters, set uniform weights
                continue
            
            # Perform k-means within this class
            if self.use_minibatch_kmeans and len(class_features) > 1000:
                kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                       random_state=self.random_state,
                                       batch_size=min(512, len(class_features)//5))
            else:
                kmeans = KMeans(n_clusters=n_clusters,
                              random_state=self.random_state,
                              n_init=10)
            
            try:
                cluster_labels = kmeans.fit_predict(class_features)
                centers = kmeans.cluster_centers_
                
                # Compute distances to assigned cluster centers
                distances = np.linalg.norm(
                    class_features - centers[cluster_labels], axis=1
                )
                
                # Normalize distances within this class
                if self.normalize_weights and len(distances) > 1:
                    min_dist, max_dist = distances.min(), distances.max()
                    if max_dist > min_dist:
                        distances = (distances - min_dist) / (max_dist - min_dist)
                
                weights[class_mask] = distances
                
            except Exception as e:
                warnings.warn(f"Failed to cluster class {class_label}: {e}")
                continue
        
        return weights
    
    def _compute_difficulty_weights_unsupervised(self, features: np.ndarray, 
                                               n_clusters: int) -> np.ndarray:
        """Compute difficulty weights using unsupervised clustering"""
        # Perform k-means on all features
        if self.use_minibatch_kmeans and len(features) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                   random_state=self.random_state,
                                   batch_size=min(2048, len(features)//10),
                                   verbose=1)
        else:
            kmeans = KMeans(n_clusters=n_clusters,
                          random_state=self.random_state,
                          n_init=10)
        
        print("Fitting K-means...")
        cluster_labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        print("Computing distances...")
        # Efficient batch-wise distance computation for large datasets
        if len(features) > 50000:
            distances = self._compute_distances_batched(features, centers, cluster_labels)
        else:
            distances = np.linalg.norm(
                features - centers[cluster_labels], axis=1
            )
        
        # Normalize distances within each cluster
        if self.normalize_weights:
            normalized_distances = np.zeros_like(distances)
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if cluster_mask.sum() > 1:
                    cluster_distances = distances[cluster_mask]
                    min_dist, max_dist = cluster_distances.min(), cluster_distances.max()
                    if max_dist > min_dist:
                        normalized_distances[cluster_mask] = (
                            cluster_distances - min_dist
                        ) / (max_dist - min_dist)
                    else:
                        normalized_distances[cluster_mask] = 0.0
            distances = normalized_distances
        
        return distances
    
    def _compute_distances_batched(self, features: np.ndarray, 
                                 centers: np.ndarray, 
                                 cluster_labels: np.ndarray,
                                 batch_size: int = 10000) -> np.ndarray:
        """Compute distances in batches to save memory"""
        distances = np.zeros(len(features))
        
        for i in tqdm(range(0, len(features), batch_size), desc="Computing distances"):
            end_idx = min(i + batch_size, len(features))
            batch_features = features[i:end_idx]
            batch_labels = cluster_labels[i:end_idx]
            
            batch_distances = np.linalg.norm(
                batch_features - centers[batch_labels], axis=1
            )
            distances[i:end_idx] = batch_distances
        
        return distances
    
    def fit(self, dataset, labels: Optional[np.ndarray] = None, 
            device: torch.device = torch.device('cpu')) -> 'CurriculumLearning':
        """
        Fit the curriculum learning model
        
        Args:
            dataset: PyTorch dataset or DataLoader
            labels: Optional labels for supervised clustering
            device: Device to run feature extraction on
        
        Returns:
            self
        """
        # Create data loader if needed
        if not isinstance(dataset, DataLoader):
            data_loader = DataLoader(dataset, batch_size=self.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=True)
        else:
            data_loader = dataset
        
        # Extract features
        print("Extracting features...")
        features = self.feature_extractor.extract_features(data_loader, device)
        self.features_ = features.numpy()
        
        # Handle labels
        if labels is not None:
            self.labels_ = labels
        elif hasattr(dataset, 'targets'):
            self.labels_ = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels_ = np.array(dataset.labels)
        else:
            self.labels_ = None
        
        # Determine number of clusters
        if self.n_clusters is None or self.auto_select_clusters:
            self.optimal_clusters_ = self._select_optimal_clusters(
                self.features_, self.labels_
            )
        else:
            self.optimal_clusters_ = self.n_clusters
        
        # Compute difficulty weights
        print("Computing difficulty weights...")
        if self.use_labels and self.labels_ is not None:
            self.weights_ = self._compute_difficulty_weights_supervised(
                self.features_, self.labels_, self.optimal_clusters_
            )
        else:
            self.weights_ = self._compute_difficulty_weights_unsupervised(
                self.features_, self.optimal_clusters_
            )
        
        print(f"Curriculum learning fitted with {self.optimal_clusters_} clusters")
        print(f"Weight statistics: min={self.weights_.min():.4f}, "
              f"max={self.weights_.max():.4f}, mean={self.weights_.mean():.4f}")
        
        return self
    
    def get_difficulty_weights(self) -> np.ndarray:
        """Get the computed difficulty weights"""
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.weights_.copy()
    
    def _compute_effective_dataset_size(self, weights: np.ndarray, temperature: float, 
                                       strategy: str = 'difficulty') -> float:
        """
        Compute the effective dataset size given weights and temperature
        
        Args:
            weights: Difficulty weights
            temperature: Temperature parameter for softmax
            strategy: 'difficulty' or 'easy'
        
        Returns:
            Effective dataset size as a percentage (0.0 to 1.0)
        """
        if strategy == 'difficulty':
            softmax_weights = torch.softmax(torch.tensor(weights) / temperature, dim=0).numpy()
        else:  # strategy == 'easy'
            softmax_weights = torch.softmax(-torch.tensor(weights) / temperature, dim=0).numpy()
        
        # Compute effective dataset size using entropy-based measure
        # Higher entropy = more uniform = larger effective size
        entropy = -np.sum(softmax_weights * np.log(softmax_weights + 1e-8))
        max_entropy = np.log(len(weights))  # Maximum possible entropy (uniform distribution)
        effective_size = entropy / max_entropy
        
        return effective_size
    
    def _find_temperature_for_effective_size(self, target_effective_size: float, 
                                           strategy: str = 'difficulty',
                                           temp_range: Tuple[float, float] = (0.1, 5.0),
                                           tolerance: float = 0.01,
                                           max_iterations: int = 50) -> float:
        """
        Use binary search to find temperature that achieves target effective dataset size
        
        Args:
            target_effective_size: Target effective dataset size (0.0 to 1.0)
            strategy: 'difficulty' or 'easy'
            temp_range: Range of temperatures to search within
            tolerance: Tolerance for convergence
            max_iterations: Maximum number of binary search iterations
        
        Returns:
            Temperature that achieves the target effective dataset size
        """
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        low_temp, high_temp = temp_range
        
        for _ in range(max_iterations):
            mid_temp = (low_temp + high_temp) / 2
            current_effective_size = self._compute_effective_dataset_size(
                self.weights_, mid_temp, strategy
            )
            
            if abs(current_effective_size - target_effective_size) < tolerance:
                return mid_temp
            
            if current_effective_size < target_effective_size:
                # Need higher temperature for larger effective size
                low_temp = mid_temp
            else:
                # Need lower temperature for smaller effective size  
                high_temp = mid_temp
        
        # Return the best approximation
        return (low_temp + high_temp) / 2
    
    def _compute_max_temperature(self, strategy: str = 'difficulty') -> float:
        """
        Compute the maximum temperature by finding temperature for theoretical max effective dataset size (1.0)
        
        Args:
            strategy: 'difficulty' or 'easy'
            
        Returns:
            Maximum temperature for theoretical max effective dataset size
        """
        return self._find_temperature_for_effective_size(
            target_effective_size=1.0,
            strategy=strategy,
            temp_range=(0.1, 10.0)  # Extended range for max temperature
        )

    def _compute_temperature(self, effective_percentage: float, 
                           strategy: str = 'difficulty') -> float:
        """
        Automatically compute temperature based on effective dataset percentage using binary search
        
        Args:
            effective_percentage: Current effective dataset percentage (0.0 to 1.0)
            strategy: 'difficulty' or 'easy'
        
        Returns:
            Computed temperature value
        """
        # Clamp effective_percentage to valid range
        effective_percentage = max(0.0, min(1.0, effective_percentage))
        
        # Use binary search to find temperature for the target effective size
        temperature = self._find_temperature_for_effective_size(
            target_effective_size=effective_percentage,
            strategy=strategy
        )
        
        return temperature

    def get_curriculum_weights(self, strategy: str = 'difficulty',
                             current_iteration: int = 0,
                             warmup_iterations: Optional[int] = None) -> np.ndarray:
        """
        Get curriculum weights for training with curriculum progression and automatic temperature
        
        Args:
            strategy: 'difficulty' (harder samples get higher weight) or 
                     'easy' (easier samples get higher weight)
            current_iteration: Current training iteration for curriculum progression
            warmup_iterations: Number of warmup iterations (uses instance default if None)
        
        Returns:
            Curriculum weights for sampling with curriculum progression
        """
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Use instance warmup_iterations if not provided
        if warmup_iterations is None:
            warmup_iterations = self.warmup_iterations
        
        weights = self.weights_.copy()
        
        # After curriculum learning stage: return uniform distribution (multiply by 0 + 1)
        if current_iteration >= warmup_iterations:
            # Uniform distribution: all samples have equal weight
            n_samples = len(weights)
            uniform_weights = np.ones(n_samples) 
            return uniform_weights
        
        # During curriculum learning stage: use temperature-based curriculum
        # Calculate curriculum progression factor and effective percentage
        progress = current_iteration / warmup_iterations if warmup_iterations > 0 else 1.0
        current_effective_percentage = (
            self.initial_effective_percentage + 
            (1.0 - self.initial_effective_percentage) * progress
        )
        
        # Automatically compute temperature based on effective dataset size
        temperature = self._compute_temperature(
            current_effective_percentage, strategy
        )
        
        # Calculate curriculum weights based on strategy
        if strategy == 'difficulty':
            # Higher difficulty = higher weight
            curriculum_weights = torch.softmax(
                torch.tensor(weights) / temperature, dim=0
            ).numpy()
        elif strategy == 'easy':
            # Lower difficulty = higher weight
            curriculum_weights = torch.softmax(
                -torch.tensor(weights) / temperature, dim=0
            ).numpy()
        else:
            raise ValueError("Strategy must be 'difficulty' or 'easy'")
        
        # Scale weights to achieve target effective percentage
        if current_effective_percentage < 1.0:
            # Set a fraction of weights to zero to reduce effective dataset size
            n_samples = len(curriculum_weights)
            n_active = int(n_samples * current_effective_percentage)
            
            # Get indices of samples to keep (highest weights for current strategy)
            active_indices = np.argpartition(curriculum_weights, -n_active)[-n_active:]
            
            # Set inactive samples to very low weight (but not zero to avoid sampling issues)
            scaled_weights = np.full_like(curriculum_weights, 1e-6)
            scaled_weights[active_indices] = curriculum_weights[active_indices]
            
            # Renormalize
            curriculum_weights = scaled_weights / scaled_weights.sum()
        
        return curriculum_weights
    
    def save_weights(self, filepath: str):
        """Save computed weights to file"""
        if self.weights_ is None:
            raise ValueError("No weights to save. Call fit() first.")
        
        np.save(filepath, {
            'weights': self.weights_,
            'n_clusters': self.optimal_clusters_,
            'features': self.features_,
            'labels': self.labels_
        })
    
    def load_weights(self, filepath: str):
        """Load weights from file"""
        data = np.load(filepath, allow_pickle=True).item()
        self.weights_ = data['weights']
        self.optimal_clusters_ = data['n_clusters']
        self.features_ = data.get('features')
        self.labels_ = data.get('labels')


# Convenience function for quick usage
def create_curriculum_learning(model: nn.Module, 
                             dataset,
                             n_clusters: Optional[int] = None,
                             use_labels: bool = False,
                             device: torch.device = torch.device('cpu'),
                             initial_effective_percentage: float = 0.3,
                             warmup_iterations: int = 1000,
                             **kwargs) -> Tuple[np.ndarray, CurriculumLearning]:
    """
    Convenience function to quickly create curriculum learning weights
    
    Args:
        model: Pretrained PyTorch model
        dataset: PyTorch dataset
        n_clusters: Number of clusters (None for auto-selection)
        use_labels: Whether to use supervised clustering
        device: Device for computation
        initial_effective_percentage: Initial percentage of dataset to use (0.1 to 1.0)
        warmup_iterations: Number of warmup iterations for curriculum progression
        **kwargs: Additional arguments for CurriculumLearning
    
    Returns:
        Tuple of (difficulty_weights, curriculum_learning_instance)
    """
    # Create feature extractor
    extractor = PretrainedModelExtractor(model)
    
    # Create curriculum learning instance
    curriculum = CurriculumLearning(
        feature_extractor=extractor,
        n_clusters=n_clusters,
        use_labels=use_labels,
        initial_effective_percentage=initial_effective_percentage,
        warmup_iterations=warmup_iterations,
        **kwargs
    )
    
    # Fit and get weights
    curriculum.fit(dataset, device=device)
    weights = curriculum.get_difficulty_weights()
    
    return weights, curriculum 


def create_curriculum_learning_with_dino(dataset,
                                       n_clusters: Optional[int] = None,
                                       use_labels: bool = False,
                                       device: torch.device = torch.device('cpu'),
                                       dino_model: str = 'dino_vits16',
                                       initial_effective_percentage: float = 0.3,
                                       warmup_iterations: int = 1000,
                                       **kwargs) -> Tuple[np.ndarray, CurriculumLearning]:
    """
    Convenience function to create curriculum learning weights using DINO as default
    
    Args:
        dataset: PyTorch dataset
        n_clusters: Number of clusters (None for auto-selection)
        use_labels: Whether to use supervised clustering
        device: Device for computation
        dino_model: DINO model variant ('dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8')
        initial_effective_percentage: Initial percentage of dataset to use (0.1 to 1.0)
        warmup_iterations: Number of warmup iterations for curriculum progression
        **kwargs: Additional arguments for CurriculumLearning
    
    Returns:
        Tuple of (difficulty_weights, curriculum_learning_instance)
    """
    import timm
    
    # Load pretrained DINO model
    model = timm.create_model(dino_model, pretrained=True, num_classes=0)
    model.eval()
    
    return create_curriculum_learning(
        model=model,
        dataset=dataset,
        n_clusters=n_clusters,
        use_labels=use_labels,
        device=device,
        initial_effective_percentage=initial_effective_percentage,
        warmup_iterations=warmup_iterations,
        **kwargs
    ) 
