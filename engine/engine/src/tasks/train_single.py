from engine.logger import make_logger
from engine.src.models.single_classifier import SingleObjectClassifier
from engine.config import Config

import logging
import os
import pandas as pd

logger = make_logger(__name__)

def train_classifier(config, task, model_type):
    """Train classification model
    
    Args:
        config: Configuration object
        task: Task to train on ('sentiment', 'employment', 'forward_guidance')
        model_type: Type of model to train ('logreg', 'svc', 'rf')
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training {model_type} classifier for task: {task}")
    
    # Load data
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "classified_statements.csv"))
    except FileNotFoundError:
        logger.error("Classified statements not found. Run extraction and classification first.")
        return None
    
    # Split data
    train_df = df[df['date'] < f'{config.last_year}-01-01']
    test_df = df[(df['date'] >= f'{config.last_year}-01-01') & 
                  (df['date'] < f'{config.last_year+1}-01-01')]
    
    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test samples")

    tasks_mapping = {
        'sentiment': ('sentiment_class', 'sentiment_summary'),
        'employment-levels': ('employment_levels_class', 'employment_summary'),
        'employment-dynamics': ('employment_dynamics_class', 'employment_summary'),
        'employment-concern': ('employment_concern_class', 'employment_summary'),
        'forward_guidance': ('guidance_class', 'forward_guidance_summary'),
        'interest_rate': ('interest_rate_class', 'interest_rate_summary'),
        'balance_sheet': ('balance_sheet_class', 'balance_sheet_summary')
    }

    if task not in tasks_mapping:
        logger.error(f"Unknown task: {task}")
        return None

    target_column, embedding_column = tasks_mapping[task]
    
    # Create and train model
    model = SingleObjectClassifier(
        config=config,
        model_type=model_type,
        target_column=target_column,
        embedding_column=embedding_column,
        stratify_column='chairman',
        cache_prefix=f"{task}_{model_type}"
    )

    model.logger.setLevel(logging.INFO)
    
    # Train model
    cv_results, best_params = model.fit(train_df, param_search=True)
    
    logger.info(f"Model trained successfully. Best parameters: {best_params}")
    logger.info(f"Cross-validation results: {cv_results}")
    
    # Evaluate on test set
    test_metrics = model.evaluate(test_df)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save model
    models_dir = os.path.join(config.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{task}_{model_type}_model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Add predictions to test data
    test_df[f"{task}_pred"] = model.predict(test_df)
    test_df[f"{task}_prob"] = model.predict_proba(test_df)[:, 1] if model.label_encoder.classes_.shape[0] == 2 else None
    
    # Save results
    results_path = os.path.join(config.output_dir, f"{task}_classification_results.csv")
    test_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    return model
