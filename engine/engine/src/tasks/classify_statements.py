from engine.src.processing.classifier import StatementClassifier
from engine.config import Config
from engine.logger import make_logger
import pandas as pd
import os

from engine.prompts.classification_prompts import (
    SENTIMENT_PROMPT,
    EMPLOYMENT_INFLATION_LEVELS_PROMPT,
    EMPLOYMENT_INFLATION_DYNAMICS_PROMPT,
    EMPLOYMENT_INFLATION_CONCERN_PROMPT,
    GUIDANCE_PROMPT,
    INTEREST_RATE_TRAJECTORY_PROMPT,
    BALANCE_SHEET_TRAJECTORY_PROMPT
)

from engine.prompts.classification_prompts import (
    Sentiment,
    EmploymentInflationLevels,
    EmploymentInflationDynamics,
    EmploymentInflationConcern,
    InterestRateTrajectory,
    BalanceSheetTrajectory,
    Guidance
)


from engine.src.models.single_classifier import SingleObjectClassifier
from engine.src.models.pairwise_classifier import LinearRankNet
from engine.src.models.evaluator import ModelEvaluator

logger = make_logger(__name__)

def classify_statements(config, task):
    """Classify extracted statements
    
    Args:
        config: Configuration object
        task: Task to process ('sentiment', 'employment', 'forward_guidance', 'all')
        
    Returns:
        Dataframe with classified statements
    """
    logger.info(f"Classifying statements for task: {task}")
    
    # Load data
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))
    except FileNotFoundError:
        logger.info("Extracted statements not found, loading original data")
        df = pd.read_csv(config.data_path)
    
    if task in ['sentiment', 'all']:
        sentiment_classifier = StatementClassifier(
            config=config,
            classification_prompt=SENTIMENT_PROMPT,
            output_schema=Sentiment
        )
        df = sentiment_classifier.process_statements_from_df(df, 'sentiment_summary', 'sentiment_class')
    
    if task in ['employment', 'all']:
        employment_levels_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_LEVELS_PROMPT,
            output_schema=EmploymentInflationLevels
        )
        df = employment_levels_classifier.process_statements_from_df(df, 'employment_summary', 'employment_levels_class')
        employment_dynamics_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_DYNAMICS_PROMPT,
            output_schema=EmploymentInflationDynamics
        )
        df = employment_dynamics_classifier.process_statements_from_df(df, 'employment_summary', 'employment_dynamics_class')
        employment_concern_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_CONCERN_PROMPT,
            output_schema=EmploymentInflationConcern
        )
        df = employment_concern_classifier.process_statements_from_df(df, 'employment_summary', 'employment_concern_class')

    if task in ['forward_guidance', 'all']:
        guidance_classifier = StatementClassifier(
            config=config,
            classification_prompt=GUIDANCE_PROMPT,
            output_schema=Guidance
        )
        df = guidance_classifier.process_statements_from_df(df, 'forward_guidance_summary', 'guidance_class')

    if task in ['interest_rate', 'all']:
        interest_rate_classifier = StatementClassifier(
            config=config,
            classification_prompt=INTEREST_RATE_TRAJECTORY_PROMPT,
            output_schema=InterestRateTrajectory
        )
        df = interest_rate_classifier.process_statements_from_df(df, 'interest_rate_summary', 'interest_rate_class')

    if task in ['balance_sheet', 'all']:
        balance_sheet_classifier = StatementClassifier(
            config=config,
            classification_prompt=BALANCE_SHEET_TRAJECTORY_PROMPT,
            output_schema=BalanceSheetTrajectory
        )
        df = balance_sheet_classifier.process_statements_from_df(df, 'balance_sheet_summary', 'balance_sheet_class')
    
    # Add chairman feature
    df['chairman'] = df['date'].apply(lambda x: 0 if x < '2014-01-01' else 1 if x < '2018-01-01' else 2)
    
    # Save processed data
    output_path = os.path.join(config.output_dir, "classified_statements.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved classified statements to {output_path}")
    
    return df