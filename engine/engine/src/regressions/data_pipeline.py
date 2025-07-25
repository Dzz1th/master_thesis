import pandas as pd 
import typing as t


def get_data():
    base_path = "/Users/dzz1th/Job/mgi/Soroka/data/pc_data/"
    employment_levels_path = base_path + "employment_level_pairwise_scores.csv"
    employment_dynamics_path = base_path + "employment_dynamics_pairwise_scores.csv"
    inflation_levels_path = base_path + "inflation_level_pairwise_scores.csv"
    inflation_dynamics_path = base_path + "inflation_dynamics_pairwise_scores.csv"
    interest_rate_path = base_path + "interest_rate_trajectory_pairwise_scores.csv"
    balance_sheet_path = base_path + "balance_sheet_trajectory_pairwise_scores.csv"
    guidance_path = base_path + "forward_guidance_guidance_pairwise_scores.csv"


    employment_levels_df = pd.read_csv(employment_levels_path)
    employment_dynamics_df = pd.read_csv(employment_dynamics_path)
    inflation_levels_df = pd.read_csv(inflation_levels_path)
    inflation_dynamics_df = pd.read_csv(inflation_dynamics_path)
    interest_rate_df = pd.read_csv(interest_rate_path)
    balance_sheet_df = pd.read_csv(balance_sheet_path)
    guidance_df = pd.read_csv(guidance_path)

    df = pd.DataFrame({
        'date': employment_levels_df['date'],
        'employment_level_score': employment_levels_df['employment_level_score'],
        'employment_dynamics_score': employment_dynamics_df['employment_dynamics_score'],
        'inflation_level_score': inflation_levels_df['inflation_level_score'],
        'inflation_dynamics_score': inflation_dynamics_df['inflation_dynamics_score'],
        'interest_rate_trajectory_score': interest_rate_df['interest_rate_trajectory_score'],
        'balance_sheet_trajectory_score': balance_sheet_df['balance_sheet_trajectory_score'],
        'forward_guidance_guidance_score': guidance_df['forward_guidance_guidance_score']
    })
    return df


def create_scores_diffs(df: pd.DataFrame, max_diff: int):
    for k in range(1, max_diff + 1):
        for col in df.columns:
            if col != 'date':
                col_base = col.replace('_score', '')
                df[f'{col_base}_diff_{k}'] = df[col].diff(k)

    return df


def get_rates(days: t.List[int] = [1, 3, 5, 10, 20, 30]):
    rates_path = "/Users/dzz1th/Job/mgi/Soroka/data/pc_data/us_yields.csv"
    rates_df = pd.read_csv(rates_path, index_col=0)
    rates_df.index = pd.to_datetime(rates_df.index)
    rates_df.columns = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']

    # Create delta rates 
    for term in rates_df.columns:
        for d in days:
            rates_df[f'{term}_{d}d'] = rates_df[term].shift(-d) - rates_df[term]

    for term in rates_df.columns:
        for d in days:
            rates_df[f'{term}_{d}d'] = rates_df[term].shift(-d) - rates_df[term]

    return rates_df
    
