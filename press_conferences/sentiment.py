import pandas as pd 
import re
from collections import defaultdict
import typing as t
import numpy as np
import os
import pickle
import asyncio
import itertools
import random
from tqdm import tqdm
from utils import TokenRateLimiter
from utils import find_all_cycles, create_graph_from_pairs, remove_cycles
import logging 
import tiktoken
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel, Field

import xgboost as xgb

from cache import set_cache
from guidance import guidance_pipeline
from employment import employment_pipeline
from summarization_augmentation import filter_statements, forward_guidance_filtering_prompt, employment_filtering_prompt
from models.models import PairwiseRankingSGD

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, mean_squared_error, roc_auc_score
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

set_cache()

rate_limiter = TokenRateLimiter(50, 2000000)

openai_key = "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"

embeddings_engine = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    openai_api_key = openai_key
)

chat = ChatOpenAI(api_key=openai_key, model="gpt-4o", temperature=0.2)

sentiment_classification_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a set of statements from a FED press conference transcript.
    Your task is to analyze this set of phrases and determine whether the FED is hawkish, dovish or neutral.
    Here are the criteria for your analysis:
        Level 1: Strongly Hawkish
        - Emphasis on Inflation Control: Clear priority on fighting inflation, even at the expense of growth or employment.
        - Aggressive Rate Hike Signals: Explicit suggestions of rapid or large interest rate increases (e.g., “We will continue raising rates aggressively until inflation is under control”).
        - Tightening Commitment: Use of firm language like “We must act decisively,” “We are committed to tightening,” with minimal hedging.
        - De-emphasis on Growth/Employment: Statements downplay economic growth or labor market concerns in favor of price stability.
        - Tone: Assertive and uncompromising, with strong, decisive language. 
          Example: "We will not hesitate to take necessary actions to control inflation."
        
        Level 2: Moderately Hawkish
        - Inflation as a Primary Concern: Inflation control is prioritized, but with recognition of economic risks.
        - Gradual Tightening Signals: Indications of continued rate hikes but at a measured pace (e.g., “We expect further rate increases in the coming months”).
        - Balanced Commitment: Phrases like “We are prepared to adjust rates as needed” show intent to tighten but with flexibility.
        - Acknowledgment of Risks: Mentions potential impact on growth but frames it as necessary to control inflation.
        - Tone: Confident but measured, with a balance of assertiveness and caution.
          Example: "While inflation remains a concern, we will proceed with caution to ensure economic stability."
        
        Level 3: Moderately Dovish
        - Focus on Economic Growth: Greater emphasis on supporting economic growth and employment.
        - Pause or Slowdown in Tightening: References to slowing or pausing rate hikes (e.g., “We need to assess the impact of prior increases”).
        - Conditional Softening: Statements like “We will be data-dependent moving forward” imply openness to less aggressive policy.
        - Recognition of Downside Risks: Concern over potential recession or market volatility is acknowledged.
        - Tone: Cautious and flexible, with language that suggests openness to change.
          Example: "We are closely monitoring economic indicators and will adjust our policies as necessary."
        
        Level 4: Strongly Dovish
        - Prioritizing Growth Over Inflation: Clear focus on supporting growth and employment, even if inflation persists.
        - Rate Cuts or Policy Easing Signals: Direct or indirect hints at rate cuts or easing (e.g., “We are prepared to provide additional support if necessary”).
        - Soft Language on Inflation: Downplaying inflation risks (e.g., “Inflation is expected to moderate on its own”).
        - Commitment to Accommodative Policy: Firm statements about maintaining accommodative monetary policy (e.g., “Rates will remain low for an extended period”).
        - Tone: Reassuring and supportive, with language that emphasizes stability and support.
          Example: "Our priority is to support economic growth and ensure a stable recovery."

    Please output your analysis and the result in form of a number between 1 and 4. You MUST Finish your answer with "Result: " and put result into hash tags like this: ##result##.

    Press Conference main statements:
    {statements}

    Your answer:
"""

async def classify_sentiment(statements):
    prompt = sentiment_classification_prompt.format(statements=statements)
    tokens_needed = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    await rate_limiter.wait_for_token_availability(tokens_needed)
    response = await chat.ainvoke(prompt)
    return response.content

async def classify_sentiment_statements(statements: t.List[str]):
    tasks = [classify_sentiment(statements[i]) for i in range(len(statements))]
    results = await tqdm_async.gather(*tasks, desc="Processing Transcripts", total=len(tasks))
    
    parsed_results = []
    for response in results:    
        result = re.search(r"Result: ##(.*)##", response)
        result = result.group(1).strip()
        result = int(result)
        parsed_results.append(result)

    return parsed_results

generate_hawkish_augmentation_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a subset of statements from a FED press conference transcript that are helping to understand whether the FED provided hawkish, dovish or neutral sentiment.
    Your task is to generate a similar set of phrases that could be attributed to the FED press conference where the FED provides more hawkish sentiment.
    Hawkish sentiment could be described by the following criteria:
        - Inflation as a Primary Concern: Inflation control is prioritized, but with recognition of economic risks or even at the expense of growth or employment if document is extremely hawkish.
        - Rate Hike Signals: Indications of continued rate hikes or even Explicit suggestions of rapid or large interest rate increases (e.g., “We will continue raising rates aggressively until inflation is under control”).
        - Tightening Commitment: Use ofs language like “We must act decisively,” “We are committed to tightening,”, “We are prepared to adjust rates as needed”
        - De-emphasis on Growth/Employment: Statements downplay economic growth or labor market concerns in favor of price stability.
        - Tone: Supportive and reassuring, with language that emphasizes stability and support.
          Example: "We will not hesitate to take necessary actions to control inflation.", "Our priority is to support economic growth and ensure a stable recovery.",
    These criteria should not be used all at once, but rather as a guide to generate a more hawkish sentiment.
    You should use the provided phrases as a reference for a style, order and context. Make these phrases more hawkish in their overall narrative, making additional emphasis on fighting inflation, raising rates and being more restrictive and cautious.
    Do not output anything else than the generated phrases.

    Original phrases:
    {phrases}

    Your answer:
"""

generate_dovish_augmentation_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a subset of statements from a FED press conference transcript that are helping to understand whether the FED provided hawkish, dovish or neutral sentiment.
    Your task is to generate a similar set of phrases that could be attributed to the FED press conference where the FED provides more dovish sentiment.
    
    Dovish sentiment could be described by the following criteria:
        - Focus on Economic Growth: Greater emphasis on supporting economic growth and employment.
        - Pause or Slowdown in Tightening: References to slowing or pausing rate hikes (e.g., “We need to assess the impact of prior increases”), or Direct or indirect hints at rate cuts or easing
        - Conditional Softening: Statements like “We will be data-dependent moving forward” imply openness to less aggressive policy.
        - Soft Language on Inflation: Downplaying inflation risks (e.g., “Inflation is expected to moderate on its own”).
        - Recognition of Downside Risks: Concern over potential recession or market volatility is acknowledged.
        - Tone: Cautious and flexible, with language that suggests openness to change.
          Example: "We are closely monitoring economic indicators and will adjust our policies as necessary."
    These criteria should not be used all at once, but rather as a guide to generate a more dovish sentiment.

    You should use the provided phrases as a reference for a style, order and context. Make these phrases more hawkish in their overall narrative, making additional emphasis on fighting inflation, raising rates and being more restrictive and cautious.
    Do not output anything else than the generated phrases.

    Original phrases:
    {phrases}

    Your answer:
"""

async def generate_hawkish_augmentation(phrases):
    prompt = generate_hawkish_augmentation_prompt.format(phrases=phrases)
    tokens_needed = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    await rate_limiter.wait_for_token_availability(tokens_needed)
    response = await chat.ainvoke(prompt)
    return response.content

async def generate_dovish_augmentation(phrases):
    prompt = generate_dovish_augmentation_prompt.format(phrases=phrases)
    tokens_needed = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    await rate_limiter.wait_for_token_availability(tokens_needed)
    response = await chat.ainvoke(prompt)
    return response.content

async def augment_dataset(documents: t.List[str], scores: t.List[int]):
    tasks = []
    dovish_instruction, hawkish_instruction = "Make the set of phrases more dovish", "Make the set of phrases more hawkish"
    for i in range(len(documents)):
        if scores[i] in (1, 2):
            tasks.append(generate_dovish_augmentation(documents[i]))
        elif scores[i] in (3, 4):
            tasks.append(generate_hawkish_augmentation(documents[i]))

    results = await tqdm_async.gather(*tasks, desc="Generating Augmentations", total=len(tasks))
    return list(zip(documents, results))

def create_sentiment_augmentations(df: pd.DataFrame):
    new_df = pd.DataFrame(columns=['date', 'sentiment_summary', 'sentiment_class', 'original'])
    texts = df['sentiment_summary'].tolist()
    augmented_texts = asyncio.run(augment_dataset(texts, df['sentiment_class'].tolist()))

    dates, summaries, classes, original = [], [], [], []

    for i, (date, sentiment_summary, sentiment_class) in enumerate(zip(df['date'], df['sentiment_summary'], df['sentiment_class'])):
        augmented_text = augmented_texts[i][1]
        augmented_class = 4 if sentiment_class in (1, 2) else 1  # exact score doesn't matter until we binarize the data
        dates.extend([date, date])
        summaries.extend([sentiment_summary, augmented_text])
        classes.extend([sentiment_class, augmented_class])
        original.extend([True, False])
    
    new_df = pd.DataFrame({'date': dates, 'sentiment_summary': summaries, 'sentiment_class': classes, 'original': original})
    return new_df

logging.basicConfig(level=logging.INFO)

def metric_(true_labels, predicted_labels, predicted_probabilities):
    """
        Compute how many incorrectly predicted labels were predicted confidently and not confidently.
    """

    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    predicted_confidences = np.abs(predicted_probabilities[incorrect_indices, predicted_labels[incorrect_indices]] - 0.5)
    
    # Find the number of confident incorrect predictions
    confident_incorrect_count = np.sum(predicted_confidences > 0.2)
    confident_incorrect_percentage = confident_incorrect_count / len(incorrect_indices)
    inconfident_incorrect_percentage = 1 - confident_incorrect_percentage

    logging.info(f"Confident Errors Pct: {confident_incorrect_percentage}")
    logging.info(f"Inconfident Error Pct: {inconfident_incorrect_percentage}")

def classification_pipeline():
    df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv")

    logging.info("Classifying sentiment")
    labels = asyncio.run(classify_sentiment_statements(df['sentiment_summary'].tolist()))

    train_texts = df['sentiment_summary'].tolist()[15:60]
    test_texts = df['sentiment_summary'].tolist()[60:]
    train_labels = labels[15:60]
    test_labels = labels[60:]

    logging.info("Augmenting dataset")
    train_augmentations = asyncio.run(augment_dataset(train_texts, train_labels))
    test_augmentations = asyncio.run(augment_dataset(test_texts, test_labels)) 

    #Shrink train augmentations
    train_augmentations = list(train_augmentations)
    train_augmentations[1] = train_augmentations[1][:10]
    train_augmentations[0] = train_augmentations[0][:20]
    
    # Shrink test augmentations
    test_augmentations = list(test_augmentations)
    test_augmentations[0] = test_augmentations[0][:1]
    test_augmentations[1] = test_augmentations[1][:1]

    train_texts = train_texts + train_augmentations[0] + train_augmentations[1]
    test_texts = test_texts + test_augmentations[0] + test_augmentations[1]

    train_labels = train_labels + [1] * len(train_augmentations[0]) + [4] * len(train_augmentations[1])
    test_labels = test_labels + [1] * len(test_augmentations[0]) + [4] * len(test_augmentations[1])

    # 0 - hawkish, 1 - dovish
    train_labels = np.array([0 if label in (1, 2) else 1 for label in train_labels])
    test_labels = np.array([0 if label in (1, 2) else 1 for label in test_labels])

    logging.info("Embedding dataset")
    train_embeddings = embeddings_engine.embed_documents(train_texts)
    test_embeddings = embeddings_engine.embed_documents(test_texts)

    train_embeddings = np.array(train_embeddings)
    test_embeddings = np.array(test_embeddings)

    logging.info("Training model")
    model = LogisticRegression(penalty='l2')
    grid = GridSearchCV(model, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    grid.fit(train_embeddings, train_labels)

    logging.info(f"Best parameters: {grid.best_params_}")
    logging.info(f"Best score: {grid.best_score_}")
    logging.info(f"Grid results: {grid.cv_results_}")


    logging.info("Evaluating model")
    best_model = LogisticRegression(penalty='l2', C=grid.best_params_['C'])
    best_model.fit(train_embeddings, train_labels)
    test_predictions = best_model.predict(test_embeddings)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)

    test_scores = best_model.predict_proba(test_embeddings)

    logging.info(f"Test accuracy: {test_accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    metric_(test_predictions, test_labels, test_scores)
    logging.info(f"Confusion matrix: {confusion_matrix(test_labels, test_predictions)}")

ranking_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a set of statements from a FED press conference transcript that are helping to understand whether the FED provided hawkish or dovish sentiment.
    You are also given a set of statements from another FED press conference.
    Your task is to determine which of the two press conferences is more hawkish or dovish.
    Think carefully before answering. Output your reasoning and the result ranking as 1 if the first document is more hawkish (means that the second document is more dovish) or -1 if the second document is more hawkish (means that the first document is more dovish).

    Press Conference 1:
    {statements1}

    Press Conference 2:
    {statements2}

"""

class RankingResponse(BaseModel):
    reasoning: str = Field(description="The reasoning for the ranking")
    result: t.Literal[1, -1] = Field(description="The ranking result")

ranking_chat = ChatOpenAI(api_key=openai_key, model="gpt-4o", temperature=0.2).with_structured_output(RankingResponse)

async def rank_pair(statements1, statements2, limit_rates = False):
    prompt = ranking_prompt.format(statements1=statements1, statements2=statements2)
    tokens_needed = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    if limit_rates:
        await rate_limiter.wait_for_token_availability(tokens_needed)
    try:
        response = await ranking_chat.ainvoke(prompt)
        return response.result
    except Exception as e:
        print(f"Error ranking pair: {e}") 
        return 1 #We will return 0 when we will have a model with multiple outputs
    
async def rank_press_conferences(pairs: t.List[tuple[str, str]], limit_rates = False) -> t.List[str]:
    tasks = [rank_pair(pair[0], pair[1], limit_rates) for pair in pairs]
    results = await tqdm_async.gather(*tasks, desc="Ranking Press Conferences", total=len(tasks))
    return results

def get_chairman_feature_onehot(text, df):
    """
    Extremely Inefficient
    """
    if df[df['sentiment_summary'] == text].index[0] < 12:
        return [1, 0, 0]
    elif df[df['sentiment_summary'] == text].index[0] < 28:
        return [0, 1, 0]
    else:
        return [0, 0, 1]
    
def get_employment_score(text, df):
    """
        Get employment score for a given text, based on its index in the dataframe and computed scores.
    """
    score = df[df['sentiment_summary'] == text]['employment_score'].values[0]
    return score

def get_guidance_score(text, df):
    score =  df[df['sentiment_summary'] == text]['guidance_score'].values[0]
    return [score]

def compute_embeddings(texts, serialization_name = None, forget_embeddings = False):
    ##TODO: Refactor this function to use a single function for all embeddings
    if forget_embeddings:
        embeddings = embeddings_engine.embed_documents(texts)
        with open(f"/Users/dzz1th/Job/mgi/Soroka/data/qa_data/{serialization_name}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings
    elif serialization_name is not None:
        if os.path.exists(f"/Users/dzz1th/Job/mgi/Soroka/data/qa_data/{serialization_name}_embeddings.pkl"):    
            with open(f"/Users/dzz1th/Job/mgi/Soroka/data/qa_data/{serialization_name}_embeddings.pkl", "rb") as f:
                embeddings = pickle.load(f)
                return embeddings
        else:
            embeddings = embeddings_engine.embed_documents(texts)
            with open(f"/Users/dzz1th/Job/mgi/Soroka/data/qa_data/{serialization_name}_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)
            return embeddings
    else:
        embeddings = embeddings_engine.embed_documents(texts)
        return embeddings
    


def get_ranking_data(df: pd.DataFrame, last_year: int, base_year: int = 2018):
    def sample_pairs(pairs, size):
        sample_idx = np.random.choice(len(pairs), size=size, replace=False)
        return [pairs[i] for i in sample_idx]
        
    logging.info("Ranking press conferences")

    delta = last_year - 2018 #We start our sample from 2018

    year_str = f"{last_year}-01-01"
    base_train_texts = df[df['date'] < f'{base_year}-01-01']['sentiment_summary'].tolist()
    new_train_texts = df[(df['date'] >= f'{base_year}-01-01') & (df['date'] < year_str)]['sentiment_summary'].tolist()
    test_texts = df[(df['date'] >= year_str) & (df['date'] < f'{last_year+1}-01-01')]['sentiment_summary'].tolist()

    base_train_pairs = list(itertools.combinations(base_train_texts, 2))
    new_train_pairs += list(itertools.combinations(new_train_texts, 2)) 
    test_pairs = list(itertools.combinations(test_texts, 2))

    new_plus_od_train_pairs = []
    for i in range(len(new_train_texts)):
        for j in range(len(base_train_texts)):
            new_plus_od_train_pairs.append((new_train_texts[i], base_train_texts[j]))

    new_train_pairs += sample_pairs(new_plus_od_train_pairs, delta * 100)

    base_train_pairs = sample_pairs(base_train_pairs, 400)
    train_pairs = base_train_pairs + new_train_pairs

    limit_rates = False
    train_ranks = asyncio.run(rank_press_conferences(train_pairs, limit_rates))
    test_ranks = asyncio.run(rank_press_conferences(test_pairs, limit_rates))
    return train_pairs, train_ranks, test_pairs, test_ranks


def get_embeddings(pairs, serialization_name, forget_embeddings = False):
    first_component_emb = compute_embeddings([pair[0] for pair in pairs], "first_" + serialization_name, forget_embeddings)
    second_component_emb = compute_embeddings([pair[1] for pair in pairs], "second_" + serialization_name, forget_embeddings)

    return np.array(first_component_emb), np.array(second_component_emb)

def create_features(df: pd.DataFrame, pairs: t.List[tuple[str, str]]):
    #first_chairman_features = np.array([get_chairman_feature_onehot(pair[0], df) for pair in pairs])
    #second_chairman_features = np.array([get_chairman_feature_onehot(pair[1], df) for pair in pairs])

    first_employment_scores = np.array([get_employment_score(pair[0], df) for pair in pairs]).reshape(-1, 1)
    second_employment_scores = np.array([get_employment_score(pair[1], df) for pair in pairs]).reshape(-1, 1)

    first_guidance_scores = np.array([get_guidance_score(pair[0], df) for pair in pairs]).reshape(-1, 1)
    second_guidance_scores = np.array([get_guidance_score(pair[1], df) for pair in pairs]).reshape(-1, 1)

    first_features = np.concatenate([first_employment_scores, first_guidance_scores], axis=1)
    second_features = np.concatenate([second_employment_scores, second_guidance_scores], axis=1)
    return first_features, second_features

def data_to_xgboost_format(first_features, second_features, ranks):
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

def data_pipeline(df, pairs, embeddings_name, exclude_embeddings = False, forget_embeddings = False):
    """
        Extract features for the pairs.
        Features include:
            1. embeddings
            2. chairman features (onehot)
            3. employment score
            4. guidance score
    """
    first_features, second_features = create_features(df, pairs)

    if not exclude_embeddings:
        first_embeddings, second_embeddings = get_embeddings(pairs, embeddings_name, forget_embeddings)
        first_features = np.concatenate([first_embeddings, first_features], axis=1)
        second_features = np.concatenate([second_embeddings, second_features], axis=1)
    return first_features, second_features

def compute_errors(model, db):
    """
        Compute the errors for the validation set.
        We take +-5 as values where sigmoid is almost 1 and 0 and compute difference between +-5 
            and the predicted value from the last layer of the model (before sigmoid).
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    predictions = model.predict(db)
    first_idxs, second_idxs = [2*i for i in range(len(predictions) // 2)], [2*i + 1 for i in range(len(predictions) // 2)]
    differences = predictions[first_idxs] - predictions[second_idxs]

    labels = db.get_label()
    errors = [1 - sigmoid(diff) if label == 1 else -sigmoid(diff) for diff, label in zip(differences, labels)]
    return errors

def prune_cycles(pairs, ranks):
    graph, num_docs, text_to_index = create_graph_from_pairs(pairs, ranks)
    cycles = find_all_cycles(graph, num_docs) 
    remove_idxs = remove_cycles(cycles)

    new_pairs = []
    new_ranks = []
    for i, pair in enumerate(pairs):
        if text_to_index[pair[0]] not in remove_idxs and text_to_index[pair[1]] not in remove_idxs:
            new_pairs.append(pair)
            new_ranks.append(ranks[i])

    return new_pairs, new_ranks

def train_second_stage_model(pairs_diffs: np.ndarray, xgb_diffs: np.ndarray, second_stage_y: np.ndarray) -> np.ndarray:
    """
        Train linear boosting model with a pairwise logistic loss function. 

        Args:
            pairs_diffs: List of embeddings differences (n, embedding_dim)
            xgb_diffs: Array of shape (n, 1) with the predicted s_i - s_j from the first stage model
            second_stage_y: Array of shape (n, 1) with the labels, +1 if pair[0] > pair[1], 0 otherwise

        Returns:
            w: Array of shape (embedding_dim, 1) with the weights of the linear model
    """
    embedding_dim = 3072

    def pairwise_logistic_loss_and_gradient(w, pairs_diffs, xgb_diffs, second_stage_y):
        loss = 0.0
        grad = np.zeros_like(w)

        for i, pair_diff in enumerate(pairs_diffs):
            #xgb_scores[i] - xgb_scores[j] + (np.dot(w, pairs_embeddings[i][0]) - np.dot(w, pairs_embeddings[i][1]))
            delta_s = xgb_diffs[i] + np.dot(w, (pair_diff))
            
            if second_stage_y[i] != 1: #If the pair is not correct, we invert the sign of the difference
                delta_s = -delta_s
                pair_diff = -pair_diff

            prob = 1 / (1 + np.exp(-delta_s))  # Sigmoid(delta_s)
            loss += -np.log(prob)

            grad += (prob - 1) * (pair_diff)

        loss /= len(pairs_diffs)
        grad /= len(pairs_diffs)

        return loss, grad
    
    w = np.zeros(embedding_dim)  # Initialize weights
    learning_rate = 0.01
    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0
        loss, grad = pairwise_logistic_loss_and_gradient(w, pairs_diffs, xgb_diffs, second_stage_y)
        w -= learning_rate * grad
        epoch_loss += loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(pairs_diffs):.4f}")

    return w

def text_to_index_pairs(pairs, df):
    index_pairs = []
    for pair in pairs:
        index_pairs.append((df[df['sentiment_summary'] == pair[0]].index[0], df[df['sentiment_summary'] == pair[1]].index[0])) 
    return index_pairs

def sentiment_pipeline(last_year: int = 2024, base_year: int = 2018):
    np.random.seed(42)
    random.seed(42)

    logging.info("Embedding pairs")
    df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data.csv")

    sentiment_classes = asyncio.run(classify_sentiment_statements(df['sentiment_summary'].tolist()))
    df['sentiment_class'] = sentiment_classes

    ## Create Sentiment Augmentations and Employment/Guidance Summaries
    df = create_sentiment_augmentations(df)
    guidance = asyncio.run(tqdm_async.gather(*[filter_statements(forward_guidance_filtering_prompt, text) for text in df['sentiment_summary'].tolist()]))
    df['forward_guidance_summary'] = guidance

    employment = asyncio.run(tqdm_async.gather(*[filter_statements(employment_filtering_prompt, text) for text in df['sentiment_summary'].tolist()]))
    df['employment_summary'] = employment

    df['chairman'] = df['date'].apply(lambda x: 0 if x < '2014-01-01' else 1 if x < '2018-01-01' else 2)

    train_df = df[df['date'] < f'{last_year}-01-01']
    test_df = df[(df['date'] >= f'{last_year}-01-01') & (df['date'] < f'{last_year+1}-01-01')]

    train_texts = train_df['sentiment_summary'].tolist()
    test_texts = test_df['sentiment_summary'].tolist()
    
    #Get guidance and employment scores for the train (until last year) and test [last_year, last_year + 1]

    guidance_train_scores, guidance_test_scores = guidance_pipeline(df, last_year, base_year)
    employment_train_scores, employment_test_scores = employment_pipeline(df, last_year, base_year)

    train_df['guidance_score'], test_df['guidance_score'] = guidance_train_scores, guidance_test_scores
    train_df['employment_score'], test_df['employment_score'] = employment_train_scores, employment_test_scores
    # df['guidance_score'] = np.concatenate([guidance_train_scores, guidance_test_scores, [None] * (len(df) - len(guidance_train_scores) - len(guidance_val_scores) - len(guidance_test_scores))])
    # df['employment_score'] = np.concatenate([employment_train_scores, employment_test_scores, [None] * (len(df) - len(employment_train_scores) - len(employment_val_scores) - len(employment_test_scores))])

    limit_rates = False
    test_pairs = list(itertools.combinations(test_texts, 2))
    test_ranks = asyncio.run(rank_press_conferences(test_pairs, limit_rates))

    augmentation_groups = sum([[i, i] for i in range(len(train_texts)//2)], [])
    train_chairman_groups = train_df['chairman'].tolist()

    second_stage_X = []
    second_stage_y = []
    second_stage_xgb_diffs = []

    stf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(stf.split(train_texts, train_chairman_groups, augmentation_groups)):
        fold_train_texts = [train_texts[i] for i in train_idx]
        val_texts = [train_texts[i] for i in val_idx] 
        train_pairs = list(itertools.combinations(fold_train_texts, 2))
        val_pairs = list(itertools.combinations(val_texts, 2))

        train_pairs = random.sample(train_pairs, 1000)
        train_ranks = asyncio.run(rank_press_conferences(train_pairs, limit_rates))
        val_ranks = asyncio.run(rank_press_conferences(val_pairs, limit_rates))

        ## Find Cycles in the train set and prune them
        train_pairs, train_ranks = prune_cycles(train_pairs, train_ranks)
        val_pairs, val_ranks = prune_cycles(val_pairs, val_ranks)

        train_first_features, train_second_features = data_pipeline(train_df, train_pairs, "train_embeddings", exclude_embeddings=True, forget_embeddings=True)
        val_first_features, val_second_features = data_pipeline(train_df, val_pairs, "val_embeddings", exclude_embeddings=True, forget_embeddings=True) 

        dbtrain = data_to_xgboost_format(train_first_features, train_second_features, train_ranks)
        dbval = data_to_xgboost_format(val_first_features, val_second_features, val_ranks)

        logging.info("Training model")

        num_rounds = 50
        watchlist = [(dbtrain, 'train'), (dbval, 'eval')]

        params = {
            'max_depth': 3,
            'eval_metric': 'error',
            'objective': 'rank:pairwise'
        }

        ranker = xgb.train(params, dbtrain, num_rounds, watchlist, early_stopping_rounds=10)

        xgb_val_predictions = ranker.predict(dbval)
        first_idxs, second_idxs = [2*i for i in range(len(xgb_val_predictions) // 2)], [2*i + 1 for i in range(len(xgb_val_predictions) // 2)]
        xgb_val_differences = xgb_val_predictions[first_idxs] - xgb_val_predictions[second_idxs]
        xgb_val_probs = 1 / (1 + np.exp(- xgb_val_differences))
        val_predicted_labels = [1 if diff > 0.5 else 0 for diff in xgb_val_probs]
        val_labels = dbval.get_label()
        val_labels = [val_labels[2*i] for i in range(len(val_labels) // 2)]

        logging.info(f"Val Accuracy score for {i} Fold: {accuracy_score(val_labels, val_predicted_labels)}")
        logging.info(f"Val Confusion matrix for {i} Fold: {confusion_matrix(val_labels, val_predicted_labels)}")

        second_stage_X += val_pairs
        second_stage_xgb_diffs = np.concatenate([second_stage_xgb_diffs, xgb_val_differences])
        second_stage_y += val_labels


    second_stage_first_features, second_stage_second_features = get_embeddings(second_stage_X, "second_stage_embeddings", forget_embeddings=True)
    second_stage_diff = second_stage_first_features - second_stage_second_features

    second_stage_w = train_second_stage_model(second_stage_diff, second_stage_xgb_diffs, second_stage_y)
    second_stage_corrections = np.dot(second_stage_w, second_stage_diff.T)
    second_stage_predictions = 1 / (1 + np.exp(- (second_stage_xgb_diffs + second_stage_corrections)))
    second_stage_predictions = [1 if diff > 0.5 else 0 for diff in second_stage_predictions]

    first_stage_predictions = 1 / (1 + np.exp(- second_stage_xgb_diffs))
    first_stage_predictions = [1 if diff > 0.5 else 0 for diff in first_stage_predictions]
    logging.info(f"First Stage Aggregated Val Accuracy: {accuracy_score(second_stage_y, first_stage_predictions)}")
    logging.info(f"Second Stage Training Accuracy: {accuracy_score(second_stage_y, second_stage_predictions)}")

    logging.info(f"First Stage Aggregated Val Confusion Matrix: {confusion_matrix(second_stage_y, first_stage_predictions)}")
    logging.info(f"Second Stage Training Confusion Matrix: {confusion_matrix(second_stage_y, second_stage_predictions)}")
    
    
    train_pairs = list(itertools.combinations(train_texts, 2))
    #train_pairs = random.sample(train_pairs, 2000)
    train_ranks = asyncio.run(rank_press_conferences(train_pairs, limit_rates))

    train_pairs, train_ranks = prune_cycles(train_pairs, train_ranks)
    train_first_features, train_second_features = data_pipeline(train_df, train_pairs, "train_embeddings", exclude_embeddings=True, forget_embeddings=True)
    dbtrain = data_to_xgboost_format(train_first_features, train_second_features, train_ranks)

    #test_pairs, test_ranks = prune_cycles(test_pairs, test_ranks)
    index_pairs = text_to_index_pairs(test_pairs, test_df)

    test_first_features, test_second_features = data_pipeline(test_df, test_pairs, "test_embeddings", exclude_embeddings=True, forget_embeddings=True)
    dbtest = data_to_xgboost_format(test_first_features, test_second_features, test_ranks)

    best_xgb = xgb.train(params, dbtrain, num_rounds)

    xgb_test_predictions = best_xgb.predict(dbtest)
    first_idxs, second_idxs = [2*i for i in range(len(xgb_test_predictions) // 2)], [2*i + 1 for i in range(len(xgb_test_predictions) // 2)]
    xgb_test_differences = xgb_test_predictions[first_idxs] - xgb_test_predictions[second_idxs]
    xgb_test_probs = 1 / (1 + np.exp(- xgb_test_differences))

    test_first_features, test_second_features = get_embeddings(test_pairs, "test_embeddings", forget_embeddings=True)
    test_diff = test_first_features - test_second_features
    test_corrections = np.dot(second_stage_w, test_diff.T)
    test_probs_incorrected = 1 / (1 + np.exp(- xgb_test_differences))
    test_probs_corrected = 1 / (1 + np.exp(- (xgb_test_differences + test_corrections)))

    test_labels = dbtest.get_label() 
    test_labels = [test_labels[2*i] for i in range(len(test_labels) // 2)]

    incorrected_auc = roc_auc_score(test_labels, test_probs_incorrected)
    corrected_auc = roc_auc_score(test_labels, test_probs_corrected)

    print('Test AUC score incorrected: ', incorrected_auc)
    print('Test AUC score corrected: ', corrected_auc) 

    test_predictions_corrected = [1 if diff > 0.5 else 0 for diff in test_probs_corrected]
    test_predictions_incorrected = [1 if diff > 0.5 else 0 for diff in test_probs_incorrected]

    print('Test Accuracy score incorrected : ', accuracy_score(test_labels, test_predictions_incorrected))
    print('Test Confusion matrix incorrected: ', confusion_matrix(test_labels, test_predictions_incorrected))
    plt.scatter(test_probs_incorrected, test_labels)
    plt.title('Scatter Plot for incorrected predictions')
    plt.xlabel('Probability of the first document being more hawkish')
    plt.ylabel('True Label')
    plt.show()

    def simulate_removal(errors_matrix, top_k = 5):
        errors_matrix = errors_matrix.copy()
        initial_size = len(errors_matrix) ** 2 - len(errors_matrix)
        initial_errors = np.sum(errors_matrix)
        initial_accuracy = 1 - initial_errors / initial_size
        print(f"Initial Accuracy: {initial_accuracy}")

        error_contribution = np.sum(errors_matrix, axis=1)
        high_error_docs = np.argsort(-error_contribution)
        for idx in high_error_docs[:top_k]:
            errors_matrix[idx, :], errors_matrix[:, idx] = 0, 0 #Assume that this document would not exist. 
            size = initial_size - (2 * len(errors_matrix) - 2)
            errors = np.sum(errors_matrix)
            new_accuracy = 1 - errors / size
            print(f"Removing document {idx} accuracy: {new_accuracy}")
        
        return


    errors_map = np.zeros(shape=(len(test_texts), len(test_texts)))
    min_idx = min(min(index_pairs[i][0] for i in range(len(index_pairs))), min(index_pairs[i][1] for i in range(len(index_pairs))))
    for i, pred in enumerate(test_predictions_incorrected):
        if pred != test_labels[i]:
            row = index_pairs[i][0] % min_idx
            col = index_pairs[i][1] % min_idx
            errors_map[row][col], errors_map[col][row] = 1, 1

    #Because we prune cycles in the test set, we have slightly different during the removal procedure
    print('Incorrected Total Documents : ', len(test_texts), 'Min Index : ', min_idx)
    print('Errors on index : ', np.sum(errors_map, axis=1))
    print('Simulated removal of top 5 documents: ', simulate_removal(errors_map, top_k=5))

    print('Test Accuracy score corrected: ', accuracy_score(test_labels, test_predictions_corrected))
    print('Test Confusion matrix corrected: ', confusion_matrix(test_labels, test_predictions_corrected))
    plt.scatter(test_probs_corrected, test_labels)
    plt.title('Scatter Plot for corrected predictions')
    plt.xlabel('Probability of the first document being more hawkish')
    plt.ylabel('True Label')
    plt.show()

    errors_map = np.zeros(shape=(len(test_texts), len(test_texts)))
    min_idx = min(min(index_pairs[i][0] for i in range(len(index_pairs))), min(index_pairs[i][1] for i in range(len(index_pairs))))
    for i, pred in enumerate(test_predictions_corrected):
        if pred != test_labels[i]:
            row = index_pairs[i][0] % min_idx
            col = index_pairs[i][1] % min_idx
            errors_map[row][col], errors_map[col][row] = 1, 1

    print('Incorrected Total Documents : ', len(test_texts), 'Min Index : ', min_idx)
    print('Errors on index : ', np.sum(errors_map, axis=1))
    print('Simulated removal of top 5 documents: ', simulate_removal(errors_map, top_k=5))
    

    test_texts = test_df['sentiment_summary'].tolist()
    test_pairs = [(test_texts[i], test_texts[i]) for i in range(len(test_texts))]
    xgb_features, _ = data_pipeline(test_df, test_pairs, "test_pred_embeddings", exclude_embeddings=True, forget_embeddings=True)
    dbtest = xgb.DMatrix(xgb_features)
    xgb_predictions = best_xgb.predict(dbtest)

    test_embeddings, _ = get_embeddings(test_pairs, "test_pred_embeddings", forget_embeddings=True)
    adjustments = np.dot(second_stage_w, test_embeddings.T)
    scores = xgb_predictions
    if corrected_auc > incorrected_auc:
        scores += adjustments

    test_df['guidance_score'] = guidance_test_scores
    test_df['employment_score'] = employment_test_scores
    test_df['sentiment_scores'] = scores
    test_df.to_csv(f"/Users/dzz1th/Job/mgi/Soroka/data/qa_data/summarized_data_with_scores_{last_year}.csv", index=False)


if __name__ == "__main__":
    sentiment_pipeline()