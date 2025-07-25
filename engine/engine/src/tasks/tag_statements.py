import pandas as pd
import os
import asyncio
from pathlib import Path

from engine.config import Config
from engine.src.processing.tagger import StatementTagger
from engine.prompts.tagging_prompts import unified_macro_tagging_prompt
from engine.logger import make_logger

logger = make_logger(__name__)

def tag_statements(config: Config, task: str = "all") -> pd.DataFrame:
    """
    Tags statements in the input data using the StatementTagger.
    """
    data_path = Path(config.data_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if 'text' not in df.columns:
        raise ValueError("'text' column not found in the input data.")

    statements_to_tag = df['text'].dropna().tolist()

    logger.info(f"Initializing StatementTagger for task: {task}")
    tagger = StatementTagger(config, unified_macro_tagging_prompt)

    logger.info(f"Starting statement tagging for {len(statements_to_tag)} statements.")
    tagged_statements = tagger.process_statements(statements_to_tag)

    df['tagged_statement'] = tagged_statements

    output_filename = f"tagged_statements_{task}.csv"
    output_path = output_dir / output_filename
    logger.info(f"Saving tagged statements to {output_path}")
    df.to_csv(output_path, index=False)

    logger.info("Statement tagging completed.")
    return df

if __name__ == "__main__":
    class DummyConfig:
        def __init__(self):
            self.data_path = "/Users/dzz1th/Job/mgi/Soroka/data/pc_data/summarized_data.csv"
            self.output_dir = "/Users/dzz1th/Job/mgi/Soroka/data/pc_data/tagged_output"
            self.openai_key = "YOUR_OPENAI_KEY" 
            self.cache_embeddings = False

    try:
        config = DummyConfig()
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Running tag_statements task directly for testing.")
        tagged_df = asyncio.run(tag_statements(config, task="all"))
        logger.info(f"Successfully tagged {len(tagged_df)} statements.")

    except Exception as e:
        logger.error(f"Error during direct execution of tag_statements.py: {e}", exc_info=True) 