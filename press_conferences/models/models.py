import typing as t

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model._stochastic_gradient import BaseSGDClassifier

import torch
import torch.nn as nn

class TripletLinearClassifierNLayers(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(TripletLinearClassifierNLayers, self).__init__()

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, current_dim//2))
            self.layers.append(nn.Sigmoid())
            current_dim = current_dim//2
        
        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x.squeeze(-1)  # Output a scalar for each input

class TripletLinearClassifier2Layers(nn.Module):
    def __init__(self, input_dim):
        super(TripletLinearClassifier2Layers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(-1)  # Output a scalar for each input
    
class PairwiseLinearClassifier(nn.Module):
    def __init__(self, input_dim, n_layers: t.Literal[1, 2]):
        super(PairwiseLinearClassifier, self).__init__()
        
        if n_layers == 1:
            self.layer = nn.Sequential(
                nn.Linear(input_dim * 2, 1)
            )
        elif n_layers == 2:
            self.layer = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            )

    def forward(self, x):
        x = self.layer(x)
        return x.squeeze(-1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PairwiseRankingSGD(BaseSGDClassifier):
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-3, 
                 random_state=None, eta0=0.0):
        super().__init__(
            loss='log_loss',  # This won't be used, we override the loss
            learning_rate='constant',
            eta0=eta0,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        
    def _fit_pairwise(self, X, y, groups):
        """Fit the model using pairwise ranking loss with sigmoid."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.coef_ = np.zeros((1, n_features))
        self.intercept_ = np.zeros(1)
        
        # Learning rate
        lr = self.eta0
        
        for epoch in range(self.max_iter):
            total_loss = 0
            weight_updates = np.zeros_like(self.coef_)
            intercept_updates = 0
            
            # Iterate through each group
            for group in np.unique(groups):
                group_mask = groups == group
                X_group = X[group_mask]
                y_group = y[group_mask]
                
                # Generate all pairs within the group
                n_group = len(y_group)
                for i in range(n_group):
                    for j in range(i + 1, n_group):
                        if y_group[i] != y_group[j]:
                            # Compute scores
                            score_i = np.dot(X_group[i], self.coef_.T) + self.intercept_
                            score_j = np.dot(X_group[j], self.coef_.T) + self.intercept_
                            score_diff = score_i - score_j
                            
                            # Compute sigmoid and gradient
                            sig = sigmoid(label * score_diff)
                            grad = label * sig * (1 - sig)
                            
                            # Update weights
                            weight_updates += grad * (X_group[i] - X_group[j])
                            intercept_updates += grad
                            
                            # Compute loss
                            loss = -np.log(sig)
                            total_loss += loss
            
            # Apply updates
            self.coef_ += lr * weight_updates
            self.intercept_ += lr * intercept_updates
            
            # Early stopping if loss is small enough
            if total_loss < self.tol:
                break
                
        return self
    
    def fit(self, X, y):
        """Fit the model on the pairwise data. 
            X is a 3D array with shape (n_samples, 2, n_features)
            y is a 1D array with shape (n_samples) of errors """
            
        return self._fit_pairwise(X, y)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        scores = self.decision_function(X)
        return (scores > 0).astype(int)
    
    def decision_function(self, X1, X2):
        """Predict confidence score for pairs."""
        return np.dot(X1, self.coef_.T) - np.dot(X2, self.coef_.T)

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array([i // 2 for i in range(n_samples)])  # Pairs of samples
    
    # Train model
    model = PairwiseRankingSGD(learning_rate=0.01, max_iter=100)
    model.fit(X, y, groups)
    
    # Make predictions
    predictions = model.predict(X)
    print("Predictions shape:", predictions.shape)
    print("Sample predictions:", predictions[:10])