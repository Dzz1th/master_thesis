"""
Unified model training module for sentiment analysis pipeline
"""
import logging
import asyncio
import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Type

import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from engine.config import Config
from engine.src.llms.embedding_cache import EmbeddingCache
from engine.src.llms.llm_client import LLMClient

class SingleObjectClassifier:
    """Unified ML classifier that supports different models, embeddings, and classification types"""
    
    def __init__(self, 
                 config: Config,
                 model_type: str = 'logreg',
                 model_params: Optional[Dict[str, Any]] = None,
                 target_column: str = None,
                 embedding_column: str = None,
                 stratify_column: str = None,
                 group_column: str = None,
                 cache_prefix: str = None):
        """Initialize ML classifier
        
        Args:
            config: Configuration object
            model_type: Type of model to use ('logreg', 'svc', 'xgb', 'rf')
            model_params: Parameters for the model
            target_column: Column containing target values
            embedding_column: Column containing text to embed
            stratify_column: Column to use for stratification in cross-validation
            group_column: Column to use for grouping in cross-validation
            cache_prefix: Prefix for cached embeddings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_client = LLMClient(config)
        
        self.model_type = model_type
        self.model_params = model_params or self._get_default_params(model_type)
        self.target_column = target_column
        self.embedding_column = embedding_column
        self.stratify_column = stratify_column
        self.group_column = group_column
        self.cache_prefix = cache_prefix or target_column
        
        self.model = self._create_model()
        self.label_encoder = LabelEncoder()
        self.param_grid = self._get_param_grid(model_type)
    
    def _create_model(self) -> BaseEstimator:
        """Create model based on model type
        
        Returns:
            Scikit-learn estimator
        """
        if self.model_type == 'logreg':
            return LogisticRegression(**self.model_params)
        elif self.model_type == 'svc':
            return SVC(**self.model_params)
        elif self.model_type == 'rf':
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == 'xgb':
            return xgb.XGBClassifier(**self.model_params)
        else:
            self.logger.warning(f"Unknown model type '{self.model_type}', using LogisticRegression")
            return LogisticRegression(**self.model_params)
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        if model_type == 'logreg':
            return {'penalty': 'l2', 'C': 1.0, 'class_weight': 'balanced'}
        elif model_type == 'svc':
            return {'kernel': 'linear', 'C': 1.0, 'class_weight': 'balanced'}
        elif model_type == 'rf':
            return {'n_estimators': 100, 'class_weight': 'balanced'}
        elif model_type == 'xgb':
            return {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        else:
            return {}
    
    def _get_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """Get parameter grid for model
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of parameter grid
        """
        if model_type == 'logreg':
            return {'C': [0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'svc':
            return {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['linear', 'rbf']}
        elif model_type == 'rf':
            return {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
        elif model_type == 'xgb':
            return {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
        else:
            return {}
    
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
                return cached_embeddings
        
        # Generate embeddings if not cached or cache is invalid
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = await self.llm_client.get_embeddings(texts)
        
        # Cache embeddings
        if self.config.cache_embeddings:
            cache_manager.save_embeddings(full_cache_name, texts, embeddings)
        
        return embeddings
    
    def fit(self, 
            train_df: pd.DataFrame, 
            param_search: bool = True,
            cv: int = 5,
            n_jobs: int = -1) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Train model on training data
        
        Args:
            train_df: Training dataframe
            param_search: Whether to perform parameter search
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (metrics, best_params)
        """
        # Get texts and labels
        texts = train_df[self.embedding_column].tolist()
        labels = train_df[self.target_column].values
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Get embeddings
        embeddings = asyncio.run(self.get_embeddings(texts, "train"))
        
        # Get stratification and grouping if provided
        stratify = train_df[self.stratify_column].values if self.stratify_column else encoded_labels
        groups = train_df[self.group_column].values if self.group_column else None
        
        # Train model
        if param_search:
            best_params, cv_results = self._param_search(embeddings, encoded_labels, stratify, groups, cv, n_jobs)
            self.model_params.update(best_params)
            self.model = self._create_model()
        else:
            cv_results = self._cross_validate(self.model, embeddings, encoded_labels, stratify, groups, cv, n_jobs)
            best_params = self.model_params
        
        # Train final model on all data
        self.model.fit(embeddings, encoded_labels)
        
        return cv_results, best_params
    
    def _param_search(self, 
                     X: np.ndarray, 
                     y: np.ndarray, 
                     stratify: np.ndarray,
                     groups: Optional[np.ndarray],
                     cv: int,
                     n_jobs: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform parameter search
        
        Args:
            X: Feature matrix
            y: Target vector
            stratify: Values to stratify by
            groups: Group labels for grouped cross-validation
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_params, cv_results)
        """
        self.logger.info(f"Performing parameter search for {self.model_type}")
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Create grid search
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=cv_strategy,
            n_jobs=n_jobs,
            scoring='accuracy',
            return_train_score=True,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Get best parameters and results
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Compute additional metrics on best model
        cv_results = self._cross_validate(best_estimator, X, y, stratify, groups, cv, n_jobs)
        
        return best_params, cv_results
    
    def _cross_validate(self, 
                       estimator: BaseEstimator, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       stratify: np.ndarray,
                       groups: Optional[np.ndarray],
                       cv: int,
                       n_jobs: int) -> Dict[str, float]:
        """Perform cross-validation
        
        Args:
            estimator: Model to validate
            X: Feature matrix
            y: Target vector
            stratify: Values to stratify by
            groups: Group labels for grouped cross-validation
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary of cross-validation metrics
        """
        self.logger.info("Performing cross-validation")
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Define metrics to compute
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Perform cross-validation
        cv_results = cross_validate(
            estimator,
            X,
            y,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        # Compute mean metrics
        metrics = {}
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            metrics[f'mean_{metric}'] = cv_results[test_key].mean()
            metrics[f'std_{metric}'] = cv_results[test_key].std()
            metrics[f'mean_train_{metric}'] = cv_results[train_key].mean()
        
        self.logger.info(f"Cross-validation metrics: {metrics}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels for data
        
        Args:
            df: Dataframe with data
            
        Returns:
            Array of predicted labels
        """
        # Get texts
        texts = df[self.embedding_column].tolist()
        
        # Get embeddings
        embeddings = asyncio.run(self.get_embeddings(texts, "predict"))
        
        # Make predictions
        predictions = self.model.predict(embeddings)
        
        # Decode predictions
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        
        return decoded_predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for data
        
        Args:
            df: Dataframe with data
            
        Returns:
            Array of predicted probabilities
        """
        # Check if model supports probability prediction
        if not hasattr(self.model, "predict_proba"):
            self.logger.warning(f"Model {self.model_type} does not support probability prediction")
            return np.ones((len(df), len(self.label_encoder.classes_)))
        
        # Get texts
        texts = df[self.embedding_column].tolist()
        
        # Get embeddings
        embeddings = asyncio.run(self.get_embeddings(texts, "predict_proba"))
        
        # Make predictions
        probabilities = self.model.predict_proba(embeddings)
        
        return probabilities
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data
        
        Args:
            test_df: Test dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get true labels
        true_labels = test_df[self.target_column].values
        encoded_true = self.label_encoder.transform(true_labels)
        
        # Get predictions
        predictions = self.predict(test_df)
        encoded_predictions = self.label_encoder.transform(predictions)
        
        # Compute metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        metrics['precision'] = precision_score(true_labels, predictions, average='macro')
        metrics['recall'] = recall_score(true_labels, predictions, average='macro')
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        self.logger.info(f"Confusion matrix:\n{cm}")
        
        # Compute ROC AUC if binary classification
        if len(self.label_encoder.classes_) == 2 and hasattr(self.model, "predict_proba"):
            probabilities = self.predict_proba(test_df)[:, 1]
            metrics['roc_auc'] = roc_auc_score(encoded_true, probabilities)
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create save data
        save_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'target_column': self.target_column,
            'embedding_column': self.embedding_column,
            'stratify_column': self.stratify_column,
            'group_column': self.group_column,
            'cache_prefix': self.cache_prefix
        }
        
        # Save data
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: Config) -> 'SingleObjectClassifier':
        """Load model from disk
        
        Args:
            path: Path to load model from
            config: Configuration object
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create model
        model = cls(
            config=config,
            model_type=save_data['model_type'],
            model_params=save_data['model_params'],
            target_column=save_data['target_column'],
            embedding_column=save_data['embedding_column'],
            stratify_column=save_data['stratify_column'],
            group_column=save_data['group_column'],
            cache_prefix=save_data['cache_prefix']
        )
        
        # Restore model and label encoder
        model.model = save_data['model']
        model.label_encoder = save_data['label_encoder']
        
        return model


