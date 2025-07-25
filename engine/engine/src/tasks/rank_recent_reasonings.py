from engine.logger import make_logger
from engine.src.processing.ranker import StatementRanker, RankerSummary
from engine.src.processing.base import StatementProcessor
from engine.config import Config

from engine.prompts.ranking_prompts import (
    EMPLOYMENT_LEVEL_RANKING_PROMPT,
    EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
    INFLATION_LEVEL_RANKING_PROMPT,
    INFLATION_DYNAMICS_RANKING_PROMPT,
    FORWARD_GUIDANCE_RANKING_PROMPT,
    INTEREST_RATE_PROJECTION_RANKING_PROMPT,
    BALANCE_SHEET_PROJECTION_RANKING_PROMPT,
    REASONING_SYNTHESIS_PROMPT
)

# Assuming 'prompts' is a top-level directory accessible in PYTHONPATH
from engine.prompts.ranking_prompts import (
    EmploymentLevelRankingResponse,
    EmploymentDynamicsRankingResponse,
    InflationLevelRankingResponse,
    InflationDynamicsRankingResponse,
    GuidanceRankingResponse,
    InterestRateProjectionRankingResponse,
    BalanceSheetProjectionRankingResponse,
    SummaryResponse
)

import pandas as pd
import os
import json
import asyncio

logger = make_logger(__name__)

# Definition of ranking_configs, copied from rank_statements.py for now
# Ideally, this would be in a shared config module
ranking_configs = {
    'employment': {
        'level': {
            'prompt': EMPLOYMENT_LEVEL_RANKING_PROMPT,
            'schema': EmploymentLevelRankingResponse,
            'column': 'employment_summary'
        },
        'dynamics': {
            'prompt': EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
            'schema': EmploymentDynamicsRankingResponse,
            'column': 'employment_summary'
        }
    },
    'inflation': {
        'level': {
            'prompt': INFLATION_LEVEL_RANKING_PROMPT,
            'schema': InflationLevelRankingResponse,
            'column': 'inflation_summary' 
        },
        'dynamics': {
            'prompt': INFLATION_DYNAMICS_RANKING_PROMPT,
            'schema': InflationDynamicsRankingResponse,
            'column': 'inflation_summary' 
        }
    },
    'forward_guidance': {
        'guidance': {
            'prompt': FORWARD_GUIDANCE_RANKING_PROMPT,
            'schema': GuidanceRankingResponse,
            'column': 'forward_guidance_summary'
        }
    },
    'interest_rate': {
        'trajectory': {
            'prompt': INTEREST_RATE_PROJECTION_RANKING_PROMPT,
            'schema': InterestRateProjectionRankingResponse,
            'column': 'interest_rate_summary'
        }
    },
    'balance_sheet': {
        'trajectory': {
            'prompt': BALANCE_SHEET_PROJECTION_RANKING_PROMPT,
            'schema': BalanceSheetProjectionRankingResponse,
            'column': 'balance_sheet_summary'
        }
    }
}

def get_recent_statement_reasonings(config: Config):
    """
    Ranks the last few statements against the most recent statement for various sub-tasks
    and extracts the reasoning for these rankings.

    Args:
        config: Configuration object.

    Returns:
        A dictionary containing the rankings and reasonings for each sub-task.
    """
    logger.info("Starting extraction of recent statement ranking reasonings.")

    # Load data
    statements_file = os.path.join(config.output_dir, "extracted_statements.csv")
    if not os.path.exists(statements_file):
        logger.error(f"Extracted statements file not found: {statements_file}")
        return {}
    
    try:
        df = pd.read_csv(statements_file)
    except Exception as e:
        logger.error(f"Error loading {statements_file}: {e}")
        return {}

    if 'date' not in df.columns:
        logger.error("'date' column not found in extracted_statements.csv. Cannot sort documents.")
        return {}

    df_sorted = df.sort_values(by='date').reset_index(drop=True)

    if len(df_sorted) < 2:
        logger.warning("Not enough documents (<2) in {statements_file} to perform recent ranking analysis.")
        return {}

    # Select up to the last 5 documents
    recent_docs_df = df_sorted.tail(min(5, len(df_sorted)))

    # This check is slightly redundant due to the one above, but ensures recent_docs_df has at least 2
    if len(recent_docs_df) < 2:
        logger.warning("After selecting recent documents, still not enough (<2) to perform ranking.")
        return {}
        
    reference_doc_row = recent_docs_df.iloc[-1]
    comparison_docs_df = recent_docs_df.iloc[:-1]

    if comparison_docs_df.empty:
        logger.info("Only one recent document available. No pairs to rank for reasoning extraction.")
        # Return a structure indicating no comparisons were made, but the process ran.
        return {"metadata": "Only one recent document found, no comparisons performed."}

    all_task_reasonings = {}

    for task_name, task_config_group in ranking_configs.items():
        all_task_reasonings[task_name] = {}
        logger.info(f"Processing rankings for task: {task_name}")
        
        for subtask_name, subtask_details in task_config_group.items():
            logger.info(f"Processing {task_name}.{subtask_name} reasonings")
            
            column_name = subtask_details['column']
            if column_name not in df.columns:
                logger.warning(f"Column '{column_name}' for task {task_name}.{subtask_name} not found in CSV. Skipping.")
                all_task_reasonings[task_name][subtask_name] = []
                continue

            reference_statement_text = str(reference_doc_row[column_name])
            reference_statement_date = str(reference_doc_row['date'])

            pairs_for_subtask = []
            doc_info_for_pairs = []

            for _, comparison_doc_row in comparison_docs_df.iterrows():
                comparison_statement_text = str(comparison_doc_row[column_name])
                comparison_statement_date = str(comparison_doc_row['date'])
                
                pairs_for_subtask.append((comparison_statement_text, reference_statement_text))
                doc_info_for_pairs.append({
                    'statement_1_date': comparison_statement_date,
                    'statement_1_text': comparison_statement_text,
                    'reference_statement_date': reference_statement_date,
                    'reference_statement_text': reference_statement_text,
                })
            
            if not pairs_for_subtask:
                logger.info(f"No pairs to rank for {task_name}.{subtask_name}.")
                all_task_reasonings[task_name][subtask_name] = []
                continue

            try:
                ranker = StatementRanker(
                    config=config,
                    ranking_prompt=subtask_details['prompt'],
                    output_schema=subtask_details['schema']
                    # window_size is not used when calling process_pairs directly
                )
                
                # process_pairs is async, so we run it in an event loop
                # It returns a list of tuples: (ranking_score, reasoning_text)
                ranking_results = asyncio.run(ranker.process_pairs(pairs_for_subtask))
            except Exception as e:
                logger.error(f"Error during ranking for {task_name}.{subtask_name}: {e}")
                all_task_reasonings[task_name][subtask_name] = [{"error": str(e)}]
                continue

            subtask_reasonings_data = []
            for i, pair_info in enumerate(doc_info_for_pairs):
                ranking, reasoning = ranking_results[i]
                subtask_reasonings_data.append({
                    **pair_info,
                    'ranking': ranking,
                    'reasoning': reasoning
                })
            
            all_task_reasonings[task_name][subtask_name] = subtask_reasonings_data
            logger.info(f"Completed {task_name}.{subtask_name}, extracted {len(subtask_reasonings_data)} reasonings.")

    # Save the results to a JSON file
    output_file_path = os.path.join(config.output_dir, "recent_statement_reasonings.json")
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(all_task_reasonings, f, indent=2)
        logger.info(f"Successfully saved recent statement reasonings to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving reasonings to {output_file_path}: {e}")

    return all_task_reasonings

if __name__ == '__main__':
    config = Config(output_dir="/Users/dzz1th/Job/mgi/Soroka/data/pc_data")
    #get_recent_statement_reasonings(config)

    processor = RankerSummary(config, REASONING_SYNTHESIS_PROMPT, SummaryResponse)
    with open("/Users/dzz1th/Job/mgi/Soroka/data/pc_data/recent_statement_reasonings.json", "r") as f:
        reasonings = f.read()

    reasonings = "Reasoning json: \n" + reasonings
    summary = asyncio.run(processor.process_summary(reasonings))

    with open(config.output_dir + "/recent_statement_reasonings_summary.json", "w") as f:
        json.dump(summary.model_dump(), f)