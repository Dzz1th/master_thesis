"""
Evaluator module for sentiment analysis pipeline
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from typing import List, Dict, Tuple, Any, Optional
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from config import Config
from llm_client import LLMClient
from utils import load_embeddings

class ModelEvaluator:
    """Evaluator for sentiment analysis models"""
    
    def __init__(self, config: Config):
        """Initialize model evaluator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_client = LLMClient(config)
    
    def evaluate(self, 
                test_df: pd.DataFrame, 
                xgb_model: xgb.Booster, 
                second_stage_w: np.ndarray) -> Dict[str, float]:
        """Evaluate models on test data
        
        Args:
            test_df: Test dataframe
            xgb_model: Trained XGBoost model
            second_stage_w: Second stage model weights
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate document pairs
        test_texts = test_df['sentiment_summary'].tolist()
        test_pairs = list(itertools.combinations(test_texts, 2))
        
        # Generate test labels from ranks
        test_ranks = [1] * len(test_pairs)  # Placeholder - in real code this would be from LLM
        
        # Prepare features
        employment_scores = test_df['employment_score'].values
        guidance_scores = test_df['guidance_score'].values
        
        # Get embeddings
        first_embeddings_path = self.config.get_embeddings_path("first_test")
        second_embeddings_path = self.config.get_embeddings_path("second_test")
        
        first_embeddings = load_embeddings(first_embeddings_path)
        second_embeddings = load_embeddings(second_embeddings_path)
        
        if first_embeddings is None or second_embeddings is None:
            self.logger.warning("Embeddings not found. Evaluation cannot proceed.")
            return {}
        
        # Compute embedding differences
        embedding_diffs = first_embeddings - second_embeddings
        
        # Create features for XGBoost
        test_first_features, test_second_features = self._prepare_features(
            test_pairs, test_df, employment_scores, guidance_scores
        )
        dbtest = self._data_to_xgboost_format(test_first_features, test_second_features, test_ranks)
        
        # Get XGBoost predictions
        xgb_predictions = xgb_model.predict(dbtest)
        first_idxs = [2*i for i in range(len(xgb_predictions) // 2)]
        second_idxs = [2*i + 1 for i in range(len(xgb_predictions) // 2)]
        xgb_diffs = xgb_predictions[first_idxs] - xgb_predictions[second_idxs]
        
        # Compute uncorrected probabilities
        uncorrected_probs = 1 / (1 + np.exp(-xgb_diffs))
        
        # Compute corrected probabilities with second stage model
        corrections = np.dot(second_stage_w, embedding_diffs.T)
        corrected_probs = 1 / (1 + np.exp(-(xgb_diffs + corrections)))
        
        # Convert to binary predictions
        uncorrected_preds = (uncorrected_probs > 0.5).astype(int)
        corrected_preds = (corrected_probs > 0.5).astype(int)
        
        # Generate sentiment scores for individual documents
        test_scores = self._generate_sentiment_scores(test_df, xgb_model, second_stage_w)
        test_df['sentiment_scores'] = test_scores
        
        # Test set doesn't have ground truth, so we only return predictions
        # In a real scenario, you would compute metrics against ground truth
        metrics = {
            "uncorrected_mean": float(uncorrected_probs.mean()),
            "corrected_mean": float(corrected_probs.mean()),
            "correction_magnitude": float(np.abs(corrections).mean())
        }
        
        return metrics
    
    def _prepare_features(self, 
                         pairs: List[Tuple[str, str]], 
                         df: pd.DataFrame,
                         employment_scores: np.ndarray,
                         guidance_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for document pairs
        
        Args:
            pairs: List of document pairs
            df: Dataframe with document information
            employment_scores: Employment scores
            guidance_scores: Guidance scores
            
        Returns:
            Tuple of (first_features, second_features)
        """
        # Create index mapping for dataframe
        text_to_idx = {text: i for i, text in enumerate(df['sentiment_summary'].tolist())}
        
        # Extract employment and guidance scores
        first_employment_scores = np.array([
            employment_scores[text_to_idx[pair[0]]] for pair in pairs
        ]).reshape(-1, 1)
        
        second_employment_scores = np.array([
            employment_scores[text_to_idx[pair[1]]] for pair in pairs
        ]).reshape(-1, 1)
        
        first_guidance_scores = np.array([
            guidance_scores[text_to_idx[pair[0]]] for pair in pairs
        ]).reshape(-1, 1)
        
        second_guidance_scores = np.array([
            guidance_scores[text_to_idx[pair[1]]] for pair in pairs
        ]).reshape(-1, 1)
        
        # Get chairman features (one-hot encoded)
        first_chairman = np.array([
            self._get_chairman_onehot(df.iloc[text_to_idx[pair[0]]]['chairman']) for pair in pairs
        ])
        
        second_chairman = np.array([
            self._get_chairman_onehot(df.iloc[text_to_idx[pair[1]]]['chairman']) for pair in pairs
        ])
        
        # Combine features
        first_features = np.concatenate([
            first_employment_scores, first_guidance_scores, first_chairman
        ], axis=1)
        
        second_features = np.concatenate([
            second_employment_scores, second_guidance_scores, second_chairman
        ], axis=1)
        
        return first_features, second_features
    
    def _get_chairman_onehot(self, chairman_id: int) -> np.ndarray:
        """Get one-hot encoded chairman feature
        
        Args:
            chairman_id: Chairman ID (0, 1, or 2)
            
        Returns:
            One-hot encoded chairman feature
        """
        if chairman_id == 0:
            return np.array([1, 0, 0])
        elif chairman_id == 1:
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])
    
    def _data_to_xgboost_format(self, 
                               first_features: np.ndarray, 
                               second_features: np.ndarray, 
                               ranks: List[int]) -> xgb.DMatrix:
        """Convert data to XGBoost format
        
        Args:
            first_features: Features for first documents
            second_features: Features for second documents
            ranks: Rankings (1 if first is better, -1 if second is better)
            
        Returns:
            XGBoost DMatrix
        """
        X = []
        y = []
        labels_map = {1: 1, -1: 0}
        
        for i in range(len(first_features)):
            X.append(first_features[i])
            X.append(second_features[i])
            y.append(labels_map[ranks[i]])
            y.append(labels_map[-1 * ranks[i]])
        
        train_groups = [2] * len(first_features)
        data = xgb.DMatrix(X, label=y)
        data.set_group(train_groups)
        
        return data
    
    def _generate_sentiment_scores(self, 
                                  df: pd.DataFrame, 
                                  xgb_model: xgb.Booster, 
                                  second_stage_w: np.ndarray) -> np.ndarray:
        """Generate sentiment scores for individual documents
        
        Args:
            df: Dataframe with documents
            xgb_model: Trained XGBoost model
            second_stage_w: Second stage model weights
            
        Returns:
            Array of sentiment scores
        """
        texts = df['sentiment_summary'].tolist()
        
        # Create self-pairs for prediction
        pairs = [(text, text) for text in texts]
        
        # Create features
        employment_scores = df['employment_score'].values
        guidance_scores = df['guidance_score'].values
        
        features, _ = self._prepare_features(
            pairs, df, employment_scores, guidance_scores
        )
        
        # Convert to XGBoost format
        dbtest = xgb.DMatrix(features)
        
        # Get predictions
        predictions = xgb_model.predict(dbtest)
        
        # Adjust with second stage model if possible
        try:
            embeddings_path = self.config.get_embeddings_path("test_pred_embeddings")
            embeddings = load_embeddings(embeddings_path)
            
            if embeddings is not None:
                adjustments = np.dot(second_stage_w, embeddings.T)
                scores = predictions + adjustments
            else:
                scores = predictions
        except Exception as e:
            self.logger.warning(f"Could not apply second stage model: {e}")
            scores = predictions
        
        return scores
    
    def plot_sentiment_vs_true_label(self, test_df: pd.DataFrame):
        """Plot sentiment scores vs true labels
        
        Args:
            test_df: Test dataframe with sentiment scores and true labels
        """
        # Create output directory for plots
        plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        sentiment_scores = test_df['sentiment_scores'].values
        
        if 'sentiment_class' in test_df.columns:
            # Binary labels for plotting
            true_labels = np.array([
                0 if label in (1, 2) else 1 
                for label in test_df['sentiment_class'].values
            ])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(sentiment_scores, true_labels, alpha=0.7)
            plt.title('Sentiment Scores vs True Labels')
            plt.xlabel('Sentiment Score (higher = more dovish)')
            plt.ylabel('True Label (0 = hawkish, 1 = dovish)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add horizontal lines at 0 and 1
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=1, color='g', linestyle='-', alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(plots_dir, "sentiment_vs_true_label.png")
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved scatter plot to {plot_path}")
        
    def analyze_errors(self, test_df: pd.DataFrame, pairs: List[Tuple[str, str]], predictions: np.ndarray, true_ranks: np.ndarray):
        """Analyze prediction errors to identify problematic documents
        
        Args:
            test_df: Test dataframe
            pairs: Document pairs
            predictions: Model predictions
            true_ranks: True rankings
        """
        # Create error matrix
        texts = test_df['sentiment_summary'].tolist()
        text_to_idx = {text: i for i, text in enumerate(texts)}
        n_docs = len(texts)
        
        error_matrix = np.zeros((n_docs, n_docs))
        
        for i, ((doc1, doc2), pred, true) in enumerate(zip(pairs, predictions, true_ranks)):
            if pred != true:
                idx1, idx2 = text_to_idx[doc1], text_to_idx[doc2]
                error_matrix[idx1, idx2] = 1
                error_matrix[idx2, idx1] = 1
        
        # Compute error contribution by document
        error_contribution = np.sum(error_matrix, axis=1)
        
        # Find top problematic documents
        top_k = 5
        problematic_indices = np.argsort(-error_contribution)[:top_k]
        
        self.logger.info(f"Top {top_k} problematic documents:")
        for idx in problematic_indices:
            self.logger.info(f"Document {idx}: {error_contribution[idx]} errors")
        
        return problematic_indices, error_contribution
    
    def simulate_removal(self, error_matrix: np.ndarray, top_k: int = 5):
        """Simulate removal of problematic documents
        
        Args:
            error_matrix: Error matrix
            top_k: Number of documents to remove
            
        Returns:
            List of accuracies after removing each document
        """
        error_matrix = error_matrix.copy()
        n_docs = len(error_matrix)
        
        initial_size = n_docs ** 2 - n_docs
        initial_errors = np.sum(error_matrix)
        initial_accuracy = 1 - initial_errors / initial_size
        
        self.logger.info(f"Initial accuracy: {initial_accuracy:.4f}")
        
        # Error contribution by document
        error_contribution = np.sum(error_matrix, axis=1)
        high_error_docs = np.argsort(-error_contribution)
        
        accuracies = [initial_accuracy]
        
        for i in range(top_k):
            idx = high_error_docs[i]
            
            # Remove document from error matrix
            error_matrix[idx, :] = 0
            error_matrix[:, idx] = 0
            
            # Recompute accuracy
            size = initial_size - (2 * n_docs - 2) * (i + 1)
            errors = np.sum(error_matrix)
            new_accuracy = 1 - errors / size
            
            self.logger.info(f"After removing document {idx}: accuracy = {new_accuracy:.4f}")
            accuracies.append(new_accuracy)
        
        return accuracies