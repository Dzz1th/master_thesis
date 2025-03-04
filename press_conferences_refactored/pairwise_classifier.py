"""
Simple single-layer model for three-way pairwise ranking
"""
import typing as t
import os
import pickle
import logging
import numpy as np
import pandas as pd
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

from config import Config
from embedding_cache import EmbeddingCache
from llm_client import LLMClient

class LinearRankNet:
    """
    Simple single-layer ranker that outputs document scores where:
    - sigmoid(s_i - s_j) ≈ 1 when x_i is ranked higher than x_j (label 1)
    - sigmoid(s_i - s_j) ≈ 0.5 when x_i and x_j are equal (label 0)
    - sigmoid(s_i - s_j) ≈ 0 when x_i is ranked lower than x_j (label -1)
    """
    
    def __init__(self, 
                 config: Config,
                 model_params: Optional[Dict[str, Any]] = None,
                 text_column: str = None,
                 feature_columns: Optional[List[str]] = None,
                 cache_prefix: str = None):
        """Initialize simple three-way ranker
        
        Args:
            config: Configuration object
            model_params: Parameters for the model
            text_column: Column containing text for embeddings
            feature_columns: Columns containing additional features
            cache_prefix: Prefix for cached embeddings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.model_params = model_params or self._get_default_params()
        self.text_column = text_column
        self.feature_columns = feature_columns or []
        self.cache_prefix = cache_prefix or "simple_three_way_ranker"
        
        self.weights = None
        self.bias = None
        self.feature_dim = None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for model
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'learning_rate': 0.01,
            'regularization': 0.01,
            'num_epochs': 200,
            'margin': 0.1,  # Margin for the zero-labeled pairs
            'batch_size': 64,
            'initialization_scale': 0.01
        }
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function
        
        Args:
            x: Input values
            
        Returns:
            Sigmoid of input. Clip for numerical stability
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    async def get_embeddings(self, texts: List[str], cache_name: str) -> np.ndarray:
        """Get embeddings for texts with smart caching
        
        Args:
            texts: List of texts to embed
            cache_name: Name for embedding cache
            
        Returns:
            Array of embeddings
        """
        # Initialize cache manager
        cache_dir = os.path.join(self.config.output_dir, "embedding_cache")
        cache_manager = EmbeddingCache(cache_dir)
        
        # Try to load embeddings from cache
        if self.config.cache_embeddings:
            full_cache_name = f"{self.cache_prefix}_{cache_name}"
            cached_embeddings = cache_manager.load_embeddings(full_cache_name, texts)
            
            if cached_embeddings is not None:
                self.logger.info(f"Cache Hit for embeddings")
                return cached_embeddings
        
        llm_client = LLMClient(self.config)
        
        # Generate embeddings if not cached or cache is invalid
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = await llm_client.get_embeddings(texts)
        
        # Cache embeddings
        if self.config.cache_embeddings:
            cache_manager.save_embeddings(full_cache_name, texts, embeddings)
        
        return embeddings
    
    def prepare_all_data(self, 
                     train_pairs: List[Tuple[str, str]], 
                     train_rankings: List[int],
                     val_pairs: List[Tuple[str, str]], 
                     val_rankings: List[int],
                     df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare all data for training and validation at once
        
        Args:
            train_pairs: List of document pairs for training
            train_rankings: List of rankings for training pairs
            val_pairs: List of document pairs for validation
            val_rankings: List of rankings for validation pairs
            df: Dataframe with additional features
            
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        self.logger.info("Preparing training and validation data")
        
        # Prepare training data
        train_data = self.prepare_data(train_pairs, train_rankings, df)
        
        # Prepare validation data
        val_data = self.prepare_data(val_pairs, val_rankings, df)
        
        return train_data, val_data
    
    def apply_pca(self, features: np.ndarray, 
                  pca_components: int,
                  mode: t.Literal['train', 'val', 'test']) -> np.ndarray:
        """Apply PCA to features"""
        if pca_components > 0:
            if mode == 'train':
                self.pca = PCA(n_components=pca_components)
                return self.pca.fit_transform(features)
            else:
                return self.pca.transform(features)
        return features
    
    def prepare_data(self, 
                    pairs: List[Tuple[str, str]], 
                    rankings: List[int],
                    df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for model
        
        Args:
            pairs: List of document pairs (text1, text2)
            rankings: List of rankings (1 if first is better, 0 if equal, -1 if second is better)
            df: Dataframe with additional features
            
        Returns:
            Dictionary with prepared data
        """
        # Get text pairs
        first_texts = [pair[0] for pair in pairs]
        second_texts = [pair[1] for pair in pairs]
        
        # Get embeddings
        first_embeddings = asyncio.run(self.get_embeddings(first_texts, "first"))
        second_embeddings = asyncio.run(self.get_embeddings(second_texts, "second"))

        # Create text to index mapping
        all_texts = sorted(set(first_texts + second_texts))
        text_to_idx = {text: i for i, text in enumerate(all_texts)}
        
        # Get indexes for each document in the pair
        first_idxs = np.array([text_to_idx[text] for text in first_texts])
        second_idxs = np.array([text_to_idx[text] for text in second_texts])
        
        # Prepare additional features if provided
        if self.feature_columns:
            # Extract features for each unique document
            idx_to_row = {i: df[df[self.text_column] == text].index[0] for i, text in enumerate(all_texts) if text in df[self.text_column].values}
            additional_features = np.zeros((len(all_texts), len(self.feature_columns)))
            
            for i, text in enumerate(all_texts):
                if i in idx_to_row:
                    additional_features[i] = df.iloc[idx_to_row[i]][self.feature_columns].values
            
            # Combine embeddings and features
            all_embeddings = np.array([first_embeddings[i] if i < len(first_embeddings) else second_embeddings[i - len(first_embeddings)] 
                                       for i in range(len(first_embeddings) + len(second_embeddings))])
                                       
            features = np.hstack([all_embeddings, additional_features])
        else:
            # Just use embeddings
            all_embeddings = {}
            for i, text in enumerate(first_texts):
                all_embeddings[text_to_idx[text]] = first_embeddings[i]
            for i, text in enumerate(second_texts):
                all_embeddings[text_to_idx[text]] = second_embeddings[i]
                
            features = np.array([all_embeddings[i] for i in range(len(all_texts))])
        
        # Store feature dimension
        self.feature_dim = features.shape[1]
        
        # Convert rankings to target differences
        # 1 -> 1 (sigmoid should be close to 1, so difference should be large positive)
        # 0 -> 0 (sigmoid should be close to 0.5, so difference should be close to 0)
        # -1 -> -1 (sigmoid should be close to 0, so difference should be large negative)
        
        return {
            'features': features,
            'first_idxs': first_idxs,
            'second_idxs': second_idxs,
            'rankings': np.array(rankings),
            'text_to_idx': text_to_idx,
            'unique_texts': all_texts,
            'num_pairs': len(pairs)
        }
    
    def _initialize_weights(self):
        """Initialize model weights"""
        if self.feature_dim is None:
            raise ValueError("Feature dimension not set")
            
        # Initialize weights with small random values
        self.weights = np.random.randn(self.feature_dim) * self.model_params['initialization_scale']
        self.bias = 0.0
    
    def _compute_document_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute scores for documents
        
        Args:
            features: Document features
            
        Returns:
            Document scores
        """
        return np.dot(features, self.weights) + self.bias
    
    def _compute_gradients(self, 
                      features: np.ndarray, 
                      first_idxs: np.ndarray, 
                      second_idxs: np.ndarray, 
                      rankings: np.ndarray, 
                      batch_indices: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for batch based on the modified loss function
        
        Args:
            features: Document features
            first_idxs: Indices of first documents in pairs
            second_idxs: Indices of second documents in pairs
            rankings: Rankings for pairs
            batch_indices: Indices for current batch
            
        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        batch_size = len(batch_indices)
        
        # Get batch data
        batch_first_idxs = first_idxs[batch_indices]
        batch_second_idxs = second_idxs[batch_indices]
        batch_rankings = rankings[batch_indices]
        
        # Compute scores for all documents
        scores = self._compute_document_scores(features)
        
        # Compute score differences for pairs in batch
        s_i = scores[batch_first_idxs]
        s_j = scores[batch_second_idxs]
        diff = s_i - s_j
        
        # Initialize gradients
        dw = np.zeros_like(self.weights)
        db = 0.0

        # Store predictions for dynamic class distribution penalty
        predictions = []
        
        # For each pair, compute gradient based on ranking
        for k in range(batch_size):
            i, j = batch_first_idxs[k], batch_second_idxs[k]
            r = batch_rankings[k]
            
            # Get scaling factors
            sigma = self.model_params['sigma']
            weight_1_vs_minus1 = self.model_params['error_weight_1_vs_minus1']
            weight_with_0 = self.model_params['error_weight_with_0']
            
            # Determine prediction based on current scores
            pred = 1 if diff[k] > self.model_params['margin'] else (
                -1 if diff[k] < -self.model_params['margin'] else 0
            )

            predictions.append(pred)
            
            class_weight = self.model_params.get('class_weights', {}).get(r, 1.0)

            if r == 1:  # First doc ranked higher (yi > yj)
                # Loss = log(1 + exp(-sigma * (si - sj)))
                exp_term = np.exp(-sigma * diff[k])
                factor = -sigma * exp_term / (1 + exp_term)  # Derivative of loss w.r.t difference
                factor *= class_weight  # Apply error weight
                
            elif r == -1:  # Second doc ranked higher (yi < yj)
                # Loss = log(1 + exp(-sigma * (sj - si))) = log(1 + exp(sigma * (si - sj)))
                exp_term = np.exp(sigma * diff[k])
                factor = sigma * exp_term / (1 + exp_term)  # Derivative of loss w.r.t difference
                factor *= class_weight  # Apply error weight
                
            else:  # r == 0, docs are equal (yi = yj)
                # Compute distance
                if abs(diff[k]) < self.model_params.get('eps', 1e-6):
                    # If scores are close enough, loss = log(2)
                    # Since log(2) is constant, gradient is zero
                    factor = 0
                else:
                    # Otherwise, margin-based squared hinge loss
                    margin = self.model_params['margin']
                    if abs(diff[k]) < margin:
                        factor = 0  # Within margin, no gradient
                    else:
                        factor = 2 * (diff[k] - np.sign(diff[k]) * margin)  # Derivative of squared hinge loss
                        factor *= class_weight  # Apply error weight
            
            # Update gradients based on features
            if factor != 0:  # Skip updates for zero gradients
                dw += factor * (features[i] - features[j])
                # Bias cancels out in the difference, so db remains 0
        
        # Average over batch
        if batch_size > 0:
            dw /= batch_size

        # Apply class distribution penalty if enabled
        if self.model_params.get('class_distribution_penalty', 0) > 0:
            predictions = np.array(predictions)
            class_counts = {
                -1: np.sum(predictions == -1),
                0: np.sum(predictions == 0),
                1: np.sum(predictions == 1)
            }
            
            # Calculate what percentage each class represents
            total = batch_size
            class_percentages = {k: v/total for k, v in class_counts.items()}
            
            # Ideal distribution would be even across classes
            ideal_percentage = 1/3
            
            # Compute deviation from ideal distribution
            deviations = {k: (v - ideal_percentage) for k, v in class_percentages.items()}
            
            # Add penalty gradient for over-represented classes
            penalty_strength = self.model_params.get('class_distribution_penalty')
            # Apply corrections based on class imbalance
            if deviations[0] > 0:  # Too many '0' predictions
                # Push scores away from the '0' region by adjusting differentials
                # This spreads scores apart making sigmoid(diff) less likely to be near 0.5
                for k in range(batch_size):
                    if abs(diff[k]) < margin:  # This pair was predicted as '0'
                        i, j = batch_first_idxs[k], batch_second_idxs[k]
                        # Push in direction of actual class or random if actual is 0
                        r = batch_rankings[k]
                        direction = r if r != 0 else (1 if np.random.random() > 0.5 else -1)
                        correction = direction * penalty_strength * deviations[0]
                        dw += correction * (features[i] - features[j]) / batch_size
            
            if deviations[1] > 0:  # Too many '1' predictions
                # Reduce the scores of first documents relative to second ones
                for k in range(batch_size):
                    if diff[k] > margin:  # This pair was predicted as '1'
                        i, j = batch_first_idxs[k], batch_second_idxs[k]
                        dw -= penalty_strength * deviations[1] * (features[i] - features[j]) / batch_size
            
            if deviations[-1] > 0:  # Too many '-1' predictions
                # Increase the scores of first documents relative to second ones
                for k in range(batch_size):
                    if diff[k] < -margin:  # This pair was predicted as '-1'
                        i, j = batch_first_idxs[k], batch_second_idxs[k]
                        dw += penalty_strength * deviations[-1] * (features[i] - features[j]) / batch_size
        
        # Add L2 regularization
        dw += self.model_params['regularization'] * self.weights
        
        return dw, db
    
    def fit(self, 
            train_data: Dict[str, Any],
            val_data: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
        """Train model on pairwise rankings using the modified loss function
        
        Args:
            pairs: List of document pairs (text1, text2)
            rankings: List of rankings (1 if first is better, 0 if equal, -1 if second is better)
            df: Dataframe with additional features
            val_pairs: Validation document pairs
            val_rankings: Validation rankings
            
        Returns:
            Dictionary with training history
        """
        # Initialize weights
        self._initialize_weights()
        
        # Training parameters
        learning_rate = self.model_params['learning_rate']
        num_epochs = self.model_params['num_epochs']
        batch_size = self.model_params['batch_size']
        num_pairs = train_data['num_pairs']
        num_batches = (num_pairs + batch_size - 1) // batch_size
        
        # Extract training data
        features = train_data['features']
        first_idxs = train_data['first_idxs']
        second_idxs = train_data['second_idxs']
        rankings_array = train_data['rankings']
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Shuffle training data
            shuffle_idxs = np.random.permutation(num_pairs)
            
            # Initialize epoch metrics
            epoch_loss = 0.0
            
            # Process mini-batches
            for batch in range(num_batches):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_pairs)
                batch_indices = shuffle_idxs[start_idx:end_idx]
                
                # Compute gradients
                dw, db = self._compute_gradients(
                    features, first_idxs, second_idxs, rankings_array, batch_indices
                )
                
                # Update weights
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            
            # Evaluate on training data
            train_metrics = self._evaluate_metrics(
                features, first_idxs, second_idxs, rankings_array
            )
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Evaluate on validation data if provided
            if val_data:
                val_metrics = self._evaluate_metrics(
                    val_data['features'],
                    val_data['first_idxs'],
                    val_data['second_idxs'],
                    val_data['rankings']
                )
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                self.logger.debug(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_metrics['loss']:.4f}, "
                                f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            else:
                self.logger.debug(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_metrics['loss']:.4f}, "
                                f"Train Acc: {train_metrics['accuracy']:.4f}")
                
            # Learning rate decay
            if (epoch + 1) % 50 == 0:
                learning_rate *= 0.5
                self.logger.debug(f"Reducing learning rate to {learning_rate}")
        
        return history
    
    def _evaluate_metrics(self, 
                     features: np.ndarray, 
                     first_idxs: np.ndarray, 
                     second_idxs: np.ndarray, 
                     rankings: np.ndarray) -> Dict[str, float]:
        """Evaluate model metrics with the modified loss function
        
        Args:
            features: Document features
            first_idxs: Indices of first documents in pairs
            second_idxs: Indices of second documents in pairs
            rankings: Rankings for pairs
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Compute scores for all documents
        scores = self._compute_document_scores(features)
        
        # Compute score differences for pairs
        s_i = scores[first_idxs]
        s_j = scores[second_idxs]
        diff = s_i - s_j
        
        # Initialize metrics
        total_loss = 0.0
        correct = 0
        
        # For each pair, compute loss and accuracy
        for k in range(len(rankings)):
            r = rankings[k]
            
            # Get scaling factors
            sigma = self.model_params['sigma']
            weight_1_vs_minus1 = self.model_params['error_weight_1_vs_minus1']
            weight_with_0 = self.model_params['error_weight_with_0']
            
            # Determine prediction based on current scores
            margin = self.model_params['margin']
            pred = 1 if diff[k] > margin else (-1 if diff[k] < -margin else 0)
            
            # Determine error weight based on true rank and prediction
            # Higher weight for confusing 1 with -1 or vice versa
            error_weight = weight_1_vs_minus1 if (r * pred == -1) else weight_with_0
            
            if r == 1:  # First doc ranked higher
                # Loss = log(1 + exp(-sigma * (si - sj)))
                base_loss = np.log1p(np.exp(-sigma * diff[k]))
                loss = error_weight * base_loss
                
            elif r == -1:  # Second doc ranked higher
                # Loss = log(1 + exp(-sigma * (sj - si))) = log(1 + exp(sigma * (si - sj)))
                base_loss = np.log1p(np.exp(sigma * diff[k]))
                loss = error_weight * base_loss
                
            else:  # r == 0, docs are equal
                # Compute distance
                if abs(diff[k]) < self.model_params.get('eps', 1e-6):
                    # If scores are close enough, loss = log(2)
                    loss = np.log(2)
                    pred = 0
                else:
                    # Otherwise, margin-based squared hinge loss
                    base_loss = max(0, abs(diff[k]) - margin) ** 2
                    loss = error_weight * base_loss
                    pred = 0 if abs(diff[k]) < margin else (1 if diff[k] > 0 else -1)
            
            # Update metrics
            total_loss += loss
            correct += (pred == r)
        
        # Average metrics
        avg_loss = total_loss / len(rankings)
        accuracy = correct / len(rankings)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def predict(self, 
               test_df: pd.DataFrame, 
               return_probs: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate scores for individual documents
        
        Args:
            test_df: Test dataframe
            return_scores: Whether to return raw scores
            
        Returns:
            Array of scores or tuple of (scores, probabilities), where probabilities are
                relative to median score.
        """
        if self.weights is None:
            raise ValueError("Model is not trained")
        
        # Get texts
        texts = test_df[self.text_column].tolist()
        
        # Get embeddings
        import asyncio
        embeddings = asyncio.run(self.get_embeddings(texts, "predict"))
        
        # Prepare features
        if self.feature_columns:
            features = np.hstack([embeddings, test_df[self.feature_columns].values])
        else:
            features = embeddings
        
        # Compute scores
        scores = self._compute_document_scores(features)
        
        # Convert to probabilities (relative to median score)
        median_score = np.median(scores)
        diffs = scores - median_score
        
        # Convert to probabilities
        # We map the difference to a probability using sigmoid
        probs = np.zeros((len(scores), 3))
        probs[:, 0] = self._sigmoid(-diffs - self.model_params['margin'])  # Prob of being worse than median
        probs[:, 1] = 1 - probs[:, 0] - probs[:, 2]  # Prob of being equal to median
        probs[:, 2] = self._sigmoid(diffs - self.model_params['margin'])   # Prob of being better than median
        
        if return_probs:
            return scores, probs
        else:
            return scores
    
    def evaluate_pairs(self, 
                      test_pairs: List[Tuple[str, str]], 
                      test_rankings: List[int],
                      test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test pairs
        
        Args:
            test_pairs: Test document pairs
            test_rankings: Test rankings
            test_df: Test dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.weights is None:
            raise ValueError("Model is not trained")
        
        # Prepare test data
        test_data = self.prepare_data(test_pairs, test_rankings, test_df)
        if self.model_params['pca_components'] > 0:
            test_data['features'] = self.apply_pca(test_data['features'], self.model_params['pca_components'], 'test')
        
        # Evaluate metrics
        metrics = self._evaluate_metrics(
            test_data['features'],
            test_data['first_idxs'],
            test_data['second_idxs'],
            test_data['rankings']
        )
        
        # Compute scores for all documents
        scores = self._compute_document_scores(test_data['features'])
        
        # Compute predicted rankings
        pred_rankings = []
        predicted_differences = []
        for i, j in zip(test_data['first_idxs'], test_data['second_idxs']):
            diff = scores[i] - scores[j]
            predicted_differences.append(diff)
            margin = self.model_params['margin']
            
            if abs(diff) < margin:
                pred_rankings.append(0)  # Equal
            elif diff > 0:
                pred_rankings.append(1)  # First is better
            else:
                pred_rankings.append(-1)  # Second is better
        
        # Compute confusion matrix
        cm = confusion_matrix(test_data['rankings'], pred_rankings, labels=[-1, 0, 1])
        
        # Log results
        self.logger.info(f"Test Loss: {metrics['loss']:.4f}, Test Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Confusion matrix:\n{cm}")
        
        # Add class-wise metrics
        metrics['confusion_matrix'] = cm
        
        # Calculate metrics for each class
        for i, class_name in enumerate(['second_better', 'equal', 'first_better']):
            label = i - 1  # -1, 0, 1
            actual_class = test_data['rankings'] == label
            pred_class = np.array(pred_rankings) == label
            
            class_accuracy = accuracy_score(actual_class, pred_class)
            metrics[f'{class_name}_accuracy'] = class_accuracy
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk
        
        Args:
            path: Path to save model
        """
        if self.weights is None:
            raise ValueError("Model is not trained")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model data
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_dim': self.feature_dim,
            'model_params': self.model_params,
            'text_column': self.text_column,
            'feature_columns': self.feature_columns,
            'cache_prefix': self.cache_prefix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {path}")

    def hyperparameter_search(self, 
                        pairs: List[Tuple[str, str]], 
                        rankings: List[int],
                        df: pd.DataFrame,
                        val_pairs: List[Tuple[str, str]],
                        val_rankings: List[int],
                        param_grid: Optional[Dict[str, List[Any]]] = None,
                        n_trials: int = 10,
                        random_seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform hyperparameter search to find the best model parameters
        
        Args:
            pairs: List of document pairs (text1, text2) for training
            rankings: List of rankings for training (1 if first is better, 0 if equal, -1 if second is better)
            df: Dataframe with additional features
            val_pairs: List of document pairs for validation
            val_rankings: List of rankings for validation
            param_grid: Dictionary of parameter grids to search, or None to use default
            n_trials: Number of parameter combinations to try
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        self.logger.info(f"Starting hyperparameter search with {n_trials} trials")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'regularization': [0.0001, 0.001, 0.01, 0.1],
                'margin': [0.1, 0.3, 0.5, 0.8],
                'sigma': [0.5, 1.0, 2.0, 5.0],
                'error_weight_1_vs_minus1': [1.0, 2.0, 3.0, 5.0],
                'error_weight_with_0': [0.5, 1.0, 1.5]
            }
        
        train_data = self.prepare_data(pairs, rankings, df)
        
        # Prepare validation data once to avoid repeating for each trial
        val_data = self.prepare_data(val_pairs, val_rankings, df)
        
        # Initialize best parameters tracking
        best_val_loss = float('inf')
        best_params = None
        best_metrics = float('-inf')
        
        # Log parameter grid
        self.logger.info(f"Parameter grid: {param_grid}")
        
        # Generate n_trials random parameter combinations
        param_keys = list(param_grid.keys())
        trials_results = []
        
        for trial in range(n_trials):
            # Sample random parameters
            current_params = self.model_params.copy()  # Start with current params
            for key in param_keys:
                current_params[key] = np.random.choice(param_grid[key])

            train_data_copy = train_data.copy()
            val_data_copy = val_data.copy()

            train_data_copy['features'] = self.apply_pca(train_data_copy['features'], 
                                                current_params['pca_components'], 
                                                'train')
            val_data_copy['features'] = self.apply_pca(val_data_copy['features'], 
                                                current_params['pca_components'], 
                                                'val')
            
            # Log current trial
            self.logger.info(f"Trial {trial+1}/{n_trials}: {current_params}")
            
            # Set model parameters for this trial
            self.model_params = current_params
            
            # Train model
            try:
                # Train model with current parameters
                history = self.fit(train_data_copy, val_data_copy)
                
                # Evaluate on validation data
                val_metrics = self._evaluate_metrics(
                    val_data_copy['features'],
                    val_data_copy['first_idxs'],
                    val_data_copy['second_idxs'],
                    val_data_copy['rankings']
                )
                
                val_loss = val_metrics['loss']
                self.logger.info(f"Validation loss: {val_loss:.4f}, accuracy: {val_metrics['accuracy']:.4f}")
                
                # Track trial results
                trial_result = {
                    'params': current_params.copy(),
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy']
                }
                trials_results.append(trial_result)
                
                #Update best parameters if found better validation loss
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_params = current_params.copy()
                #     best_metrics = val_metrics.copy()
                #     self.logger.info(f"New best parameters found with validation loss: {best_val_loss:.4f}")

                if val_metrics['accuracy'] > best_metrics:
                    best_metrics = val_metrics['accuracy']
                    best_params = current_params.copy()
                    self.logger.info(f"New best parameters found with validation accuracy: {best_metrics:.4f}")
            
            except Exception as e:
                self.logger.error(f"Error in trial {trial+1}: {e}")
                # Continue with next trial
        
        # Sort trials by validation loss
        sorted_trials = sorted(trials_results, key=lambda x: x['val_loss'])
        
        # Log best trials
        self.logger.info("Top 5 parameter combinations:")
        for i, trial in enumerate(sorted_trials[:5]):
            self.logger.info(f"Rank {i+1}: Loss={trial['val_loss']:.4f}, Accuracy={trial['val_accuracy']:.4f}, Params={trial['params']}")
        
        # Set model parameters to best found
        if best_params:
            self.model_params = best_params
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best validation accuracy: {best_metrics['accuracy']:.4f}")
            #self.logger.info(f"Best validation metrics: Loss={best_metrics['loss']:.4f}, Accuracy={best_metrics['accuracy']:.4f}")
            
            # Retrain model with best parameters
            self.fit(train_data, val_data)
        else:
            self.logger.warning("No successful trials found")
        
        return best_params, best_metrics

    def grid_search(self, 
               train_data: Dict[str, Any],
               val_data: Dict[str, Any],
               param_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform grid search to find the best model parameters
        
        Args:
            train_data: Pre-processed training data dictionary from prepare_data
            val_data: Pre-processed validation data dictionary from prepare_data
            param_grid: Dictionary of parameter grids to search
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        self.logger.info(f"Starting grid search")
        
        # Extract training data
        features = train_data['features']
        first_idxs = train_data['first_idxs']
        second_idxs = train_data['second_idxs']
        rankings_array = train_data['rankings']
        
        # Initialize best parameters tracking
        best_val_loss = float('inf')
        best_params = None
        best_metrics = None
        best_weights = None
        best_bias = None
        
        # Generate all combinations of parameters
        param_keys = sorted(param_grid.keys())
        param_values = [param_grid[key] for key in param_keys]
        
        # Count total number of combinations
        total_combinations = np.prod([len(values) for values in param_values])
        self.logger.info(f"Grid search with {total_combinations} parameter combinations")
        
        # Generate all combinations
        import itertools
        trials_results = []
        trial_count = 0
        
        for values in itertools.product(*param_values):
            trial_count += 1
            
            # Create parameter dictionary
            current_params = self.model_params.copy()  # Start with current params
            for i, key in enumerate(param_keys):
                current_params[key] = values[i]
            
            # Log current trial
            self.logger.info(f"Trial {trial_count}/{total_combinations}: {current_params}")
            
            # Set model parameters for this trial
            self.model_params = current_params
            
            # Train model
            try:
                # Initialize weights for this trial
                self._initialize_weights()
                
                # Training parameters for this trial
                learning_rate = self.model_params['learning_rate']
                num_epochs = self.model_params['num_epochs']
                batch_size = self.model_params['batch_size']
                num_pairs = train_data['num_pairs']
                num_batches = (num_pairs + batch_size - 1) // batch_size
                
                # Training loop
                for epoch in range(num_epochs):
                    # Shuffle training data
                    shuffle_idxs = np.random.permutation(num_pairs)
                    
                    # Process mini-batches
                    for batch in range(num_batches):
                        # Get batch indices
                        start_idx = batch * batch_size
                        end_idx = min(start_idx + batch_size, num_pairs)
                        batch_indices = shuffle_idxs[start_idx:end_idx]
                        
                        # Compute gradients
                        dw, db = self._compute_gradients(
                            features, first_idxs, second_idxs, rankings_array, batch_indices
                        )
                        
                        # Update weights
                        self.weights -= learning_rate * dw
                        self.bias -= learning_rate * db
                    
                    # Learning rate decay
                    if (epoch + 1) % 50 == 0:
                        learning_rate *= 0.5
                
                # Evaluate on validation data
                val_metrics = self._evaluate_metrics(
                    val_data['features'],
                    val_data['first_idxs'],
                    val_data['second_idxs'],
                    val_data['rankings']
                )
                
                val_loss = val_metrics['loss']
                self.logger.info(f"Validation loss: {val_loss:.4f}, accuracy: {val_metrics['accuracy']:.4f}, severe error rate: {val_metrics['severe_error_rate']:.4f}")
                
                # Track trial results
                trial_result = {
                    'params': current_params.copy(),
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_severe_error_rate': val_metrics['severe_error_rate'],
                    'weights': self.weights.copy(),
                    'bias': self.bias
                }
                trials_results.append(trial_result)
                
                # Update best parameters if found better validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = current_params.copy()
                    best_metrics = val_metrics.copy()
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    self.logger.info(f"New best parameters found with validation loss: {best_val_loss:.4f}")
            
            except Exception as e:
                self.logger.error(f"Error in trial {trial_count}: {e}")
                # Continue with next trial
        
        # Sort trials by validation loss
        sorted_trials = sorted(trials_results, key=lambda x: x['val_loss'])
        
        # Log best trials
        self.logger.info("Top 5 parameter combinations:")
        for i, trial in enumerate(sorted_trials[:5]):
            self.logger.info(f"Rank {i+1}: Loss={trial['val_loss']:.4f}, Accuracy={trial['val_accuracy']:.4f}, Params={trial['params']}")
        
        # Set model parameters to best found
        if best_params:
            self.model_params = best_params
            self.weights = best_weights
            self.bias = best_bias
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best validation metrics: Loss={best_metrics['loss']:.4f}, Accuracy={best_metrics['accuracy']:.4f}")
        else:
            self.logger.warning("No successful trials found")
        
        return best_params, best_metrics
    
    @classmethod
    def load(cls, path: str, config: Config) -> 'LinearRankNet':
        """Load model from disk
        
        Args:
            path: Path to load model from
            config: Configuration object
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            config=config,
            model_params=model_data['model_params'],
            text_column=model_data['text_column'],
            feature_columns=model_data['feature_columns'],
            cache_prefix=model_data['cache_prefix']
        )
        
        # Set model parameters
        model.weights = model_data['weights']
        model.bias = model_data['bias']
        model.feature_dim = model_data['feature_dim']
        
        return model