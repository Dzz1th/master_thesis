from engine.src.processing.extractor import StatementExtractor
from engine.config import Config
from engine.logger import make_logger 

from engine.prompts.summarization_prompts import (
    EMPLOYMENT_PROMPT,
    INFLATION_PROMPT,
    FORWARD_GUIDANCE_PROMPT,
    INTEREST_RATE_PROMPT,
    BALANCE_SHEET_PROMPT,
    ECONOMIC_OUTLOOK_PROMPT
)

import pandas as pd
import os 

logger = make_logger(__name__)

def extract_statements(config, task):
    """Extract statements from raw texts
    
    Args:
        config: Configuration object
        task: Task to process ('sentiment', 'employment', 'forward_guidance', 'all')
        
    Returns:
        Dataframe with extracted statements
    """
    logger.info(f"Extracting statements for task: {task}")
    
    # Load data
    df = pd.read_csv(config.data_path)
    mapping = {
        'employment': EMPLOYMENT_PROMPT,
        'inflation': INFLATION_PROMPT,
        'forward_guidance': FORWARD_GUIDANCE_PROMPT,
        'interest_rate': INTEREST_RATE_PROMPT,
        'balance_sheet': BALANCE_SHEET_PROMPT,
        'economic_outlook': ECONOMIC_OUTLOOK_PROMPT
    }
    
    tasks_to_process = mapping.keys() if task == 'all' else [task]

    for current_task in tasks_to_process:
        if current_task not in mapping:
            logger.warning(f"Task '{current_task}' not found in prompt mapping. Skipping.")
            continue

        extractor = StatementExtractor(
            config=config,
            extraction_prompt=mapping[current_task],
            chunk_size=config.chunk_size
        )

        statements_to_process = df['text'].dropna().tolist()
        extracted_statements = extractor.process_statements(statements_to_process)
        
        temp_df = pd.DataFrame({
            f'{current_task}_summary': extracted_statements,
            'original_index': df['text'].dropna().index
        })
        
        original_indices = df.index.name or 'index'
        df.reset_index(inplace=True)
        
        # Merge the results back based on index
        df = pd.merge(df, temp_df, left_on=original_indices, right_on='original_index', how='left')
        df.drop(columns=['original_index'], inplace=True)
        
        logger.info(f"Extracted {current_task} statements")

    # Save processed data
    output_path = os.path.join(config.output_dir, "extracted_statements.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved extracted statements to {output_path}")
    
    return df