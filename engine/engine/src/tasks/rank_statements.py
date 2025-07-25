from engine.logger import make_logger
from engine.src.processing.ranker import StatementRanker
from engine.config import Config
import pandas as pd
import os
import json

from engine.prompts.ranking_prompts import (
    EMPLOYMENT_LEVEL_RANKING_PROMPT,
    EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
    INFLATION_LEVEL_RANKING_PROMPT,
    INFLATION_DYNAMICS_RANKING_PROMPT,
    FORWARD_GUIDANCE_RANKING_PROMPT,
    INTEREST_RATE_PROJECTION_RANKING_PROMPT,
    BALANCE_SHEET_PROJECTION_RANKING_PROMPT
)

from engine.prompts.ranking_prompts import (
    EmploymentLevelRankingResponse,
    EmploymentDynamicsRankingResponse,
    InflationLevelRankingResponse,
    InflationDynamicsRankingResponse,
    GuidanceRankingResponse,
    InterestRateProjectionRankingResponse,
    BalanceSheetProjectionRankingResponse
)

logger = make_logger(__name__)

def rank_statements(config, task):
    """Rank statements"""
    logger.info(f"Ranking statements for task: {task}")

    # Load data
    df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))

    ranked_pairs_dir = os.path.join(config.output_dir, "ranked_pairs")
    os.makedirs(ranked_pairs_dir, exist_ok=True)

    ranking_configs = {
        'employment': {
            'level': {
                'prompt': EMPLOYMENT_LEVEL_RANKING_PROMPT,
                'schema': EmploymentLevelRankingResponse,
                'column': 'employment_summary',
                'window_size': 10
            },
            'dynamics': {
                'prompt': EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
                'schema': EmploymentDynamicsRankingResponse,
                'column': 'employment_summary',
                'window_size': 7
            }
        },
        'inflation': {
            'level': {
                'prompt': INFLATION_LEVEL_RANKING_PROMPT,
                'schema': InflationLevelRankingResponse,
                'column': 'inflation_summary',
                'window_size': 10
            },
            'dynamics': {
                'prompt': INFLATION_DYNAMICS_RANKING_PROMPT,
                'schema': InflationDynamicsRankingResponse,
                'column': 'inflation_summary',
                'window_size': 7
            }
        },
        'forward_guidance': {
            'guidance': {
                'prompt': FORWARD_GUIDANCE_RANKING_PROMPT,
                'schema': GuidanceRankingResponse,
                'column': 'forward_guidance_summary',
                'window_size': 7
            }
        },
        'interest_rate': {
            'trajectory': {
                'prompt': INTEREST_RATE_PROJECTION_RANKING_PROMPT,
                'schema': InterestRateProjectionRankingResponse,
                'column': 'interest_rate_summary',
                'window_size': 7
            }
        },
        'balance_sheet': {
            'trajectory': {
                'prompt': BALANCE_SHEET_PROJECTION_RANKING_PROMPT,
                'schema': BalanceSheetProjectionRankingResponse,
                'column': 'balance_sheet_summary',
                'window_size': 7
            }
        }
    }

    tasks_to_process = list(ranking_configs.keys()) if task == 'all' else [task]

    for current_task in tasks_to_process:
        task_configs = ranking_configs.get(current_task, {})
        for subtask, subtask_config in task_configs.items():
            logger.info(f"Processing {current_task}.{subtask} rankings")
            ranker = StatementRanker(
                config=config,
                ranking_prompt=subtask_config['prompt'],
                output_schema=subtask_config['schema'],
                window_size=subtask_config['window_size']
            )
            
            pairs, rankings, reasonings = ranker.process_pairs_from_df(df, subtask_config['column'])
            
            pairs_data = []
            for i, ((doc1, doc2), ranking) in enumerate(zip(pairs, rankings)):
                row1 = df[df[subtask_config['column']] == doc1].iloc[0]
                row2 = df[df[subtask_config['column']] == doc2].iloc[0]
                pairs_data.append({
                    'doc_1': row1[subtask_config['column']],
                    'doc_2': row2[subtask_config['column']],
                    'date_1': row1['date'],
                    'date_2': row2['date'],
                    'ranking': ranking,
                    'ranking_reasoning': reasonings[i]
                })

            output_file = os.path.join(ranked_pairs_dir, f"{current_task}_{subtask}_ranked_pairs.json")
            with open(output_file, 'w') as f:
                json.dump(pairs_data, f, indent=2)
            logger.info(f"Saved {len(pairs)} ranked pairs to {output_file}")

            
    
    return