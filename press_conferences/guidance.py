import pandas as pd 
import re
import typing as t
import numpy as np
from tqdm import tqdm
from utils import TokenRateLimiter
import logging 
import tiktoken
import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from utils import stratified_group_kfold

from cache import set_cache

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

set_cache()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('guidance_pipeline.log'),
        logging.StreamHandler()
    ]
)

rate_limiter = TokenRateLimiter(150, 40000000)

openai_key = "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"

embeddings_engine = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    openai_api_key = openai_key
)

chat = ChatOpenAI(api_key=openai_key, model="gpt-4o", temperature=0.2)

guidance_classification_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a set of statements from a FED press conference transcript.
    Your task is to analyze this set of phrases and determine whether the FED provided clear and explicit forward guidance or unclear and implicit forward guidance.
    Here are the criteria for your analysis:
        Level 1: Highly Explicit Guidance
            - Clear Time Horizons: The Fed explicitly states how long rates will remain at a certain level (e.g., “until at least next summer”).
            - Quantitative Thresholds: The Fed gives numeric triggers for policy changes (e.g., “We will not raise rates until unemployment falls below 6% and inflation hits 2% for at least six months”).
            - Commitment Language: Repeated use of unambiguous terms like “we will,” “we plan to,” “we are committed,” with minimal hedging.
        Level 2: Moderately Explicit Guidance
            - Specific Conditions but Some Flexibility: The Fed outlines the types of conditions it needs to see (e.g., “further improvements in the labor market,” “inflation returning to target range”), but doesn’t tie them to fixed numbers or an exact timeline.
            - Some Time-Related Hints: Vague references to a timeframe (e.g., “in the coming quarters,” “over the next few meetings”).
            - Conditional Statements: “We will raise rates if inflation continues to run above our target, provided the labor market remains strong,” but with some hedging words (“if,” “provided,” etc.).
        Level 3: Moderately Implicit Guidance
            - Broad Economic Criteria: The Fed points to “economic progress” or “sustained gains,” but doesn’t specify how to measure them quantitatively.
            - Heavier Use of Conditionality: Frequent phrases like “we remain data-dependent,” “too soon to tell,” “we’ll see how the economy evolves.”
            - No Clear Timeline: They might mention “future policy actions” or “coming meetings” but never commit to a particular date or sequence of moves.
        Level 4: Implicit Guidance
            - Generic Policy Statements: “We’ll adjust policy as necessary to support our mandate” without specifying any particular condition, threshold, or time horizon.
            - Minimal Detail: The Fed acknowledges it might change rates but provides no sense of whether that’s likely in the near term or far future.
            - Mostly Qualitative: Discussion focuses on the current stance (“we are at the appropriate level of restrictiveness”) without meaningful forward-looking signals.

    Please output your analysis and the result in form of a number between 1 and 4. Finish your answer with "Result: " and put result into ## .

    Press Conference main statements:
    {statements}

    Your answer:
"""

async def classify_guidance(statements):
    prompt = guidance_classification_prompt.format(statements=statements)
    tokens_needed = tiktoken.encoding_for_model("gpt-4o").encode(prompt)
    await rate_limiter.wait_for_token_availability(len(tokens_needed))
    response = await chat.ainvoke(prompt)
    return response.content

async def classify_guidance_statements(statements: t.List[str]):
    tasks = [classify_guidance(statements[i]) for i in range(len(statements))]
    results = await tqdm_async.gather(*tasks, desc="Processing Transcripts", total=len(tasks))
    
    parsed_results = []
    for response in results:    
        result = re.search(r"Result: ##(.*)##", response)
        result = result.group(1).strip()
        parsed_results.append(result)

    return parsed_results

guidance_augmentation_prompt = """
        You are an expert in macroeconomics and central bank policy.
    You are given a subset of statements from a FED press conference transcript that are helping to understand whether the FED provided explicit or implicit forward guidance.
    Your task is to generate a similar set of phrases that could be attributed to the FED press conference where the FED provides moderately explicit forward guidance.
    Moderately Explicit Guidance is defined as:
            - Specific Conditions but Some Flexibility: The Fed outlines the types of conditions it needs to see (e.g., “further improvements in the labor market,” “inflation returning to target range”), but doesn’t tie them to fixed numbers or an exact timeline.
            - Some Time-Related Hints: Vague references to a timeframe (e.g., “in the coming quarters,” “over the next few meetings”).
            - Conditional Statements: “We will raise rates if inflation continues to run above our target, provided the labor market remains strong,” but with some hedging words (“if,” “provided,” etc.).

    You should use the provided phrases as a reference for a style and order. Think about which data and events would accompany the statements with moderately explicit forward guidance.
    You should return only a list of phrases.

    Original phrases:
    {phrases}

    Your answer:
"""

async def augment_guidance(statements):
    prompt = guidance_augmentation_prompt.format(phrases=statements)
    tokens_needed = tiktoken.encoding_for_model("gpt-4o").encode(prompt)
    await rate_limiter.wait_for_token_availability(len(tokens_needed))
    response = await chat.ainvoke(prompt)
    return response.content

async def augment_guidance_statements(statements):
    tasks = [augment_guidance(statements[i]) for i in range(len(statements))]
    results = await tqdm_async.gather(*tasks, desc="Processing Transcripts", total=len(tasks))
    return results




def guidance_pipeline(data, last_year: int = 2019, base_year: int = 2018):
    """
        Guidance pipeline.

        Train model using CV on train set from 2011 to last year.
        Using CV, derive best parameters.
        After that, train model with best parameters again using CV and form predictions for each fold.
        Obtain fold predictions and put them as train data for the next stage.
        Also return predictions for test set.

        Return:
            train_scores, test_scores
    """
    print("Starting Guidance pipeline")
    classes = asyncio.run(classify_guidance_statements(data['forward_guidance_summary'].tolist()))

    data['guidance_class'] = classes
    train_data = data[data['date'] < f'{last_year}-01-01']
    test_data = data[(data['date'] >= f'{last_year}-01-01') & (data['date'] < f'{last_year+1}-01-01')]
    
    train_texts = train_data['forward_guidance_summary'].tolist()
    test_texts = test_data['forward_guidance_summary'].tolist()

    nonembed_train_texts = train_data['forward_guidance_summary'].tolist()
    nonembed_test_texts = test_data['forward_guidance_summary'].tolist()

    train_labels = train_data['guidance_class'].tolist()
    test_labels = test_data['guidance_class'].tolist()

    nonaugmented_train_labels = train_data['guidance_class'].tolist()
    nonaugmented_test_labels = test_data['guidance_class'].tolist()

    train_labels = np.array([0 if label in ('1', '2') else 1 for label in train_labels])
    test_labels = np.array([0 if label in ('1', '2') else 1 for label in test_labels])
    
    nonaugmented_train_labels = np.array([0 if label in ('1', '2') else 1 for label in nonaugmented_train_labels])
    nonaugmented_test_labels = np.array([0 if label in ('1', '2') else 1 for label in nonaugmented_test_labels])

    chairman_labels = train_data['chairman'].tolist() #because we duplicate the data with embeddings

    train_embeddings = np.array(embeddings_engine.embed_documents(train_texts))
    test_embeddings = np.array(embeddings_engine.embed_documents(test_texts))

    nonaugmented_train_embeddings = np.array(embeddings_engine.embed_documents(nonembed_train_texts))
    nonaugmented_test_embeddings = np.array(embeddings_engine.embed_documents(nonembed_test_texts))

    grid = ParameterGrid({'C': [0.1, 1, 10, 100]})
    grid_results = {tuple(params.items()): [] for params in grid}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    #Stratify on chairman labels and ensure that augmentations of the train do not get mixed into the validation set
    for train_index, test_index in skf.split(train_embeddings, chairman_labels):
        X_train, X_test = train_embeddings[train_index], train_embeddings[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        for params in grid:
            model = SVC(**dict(params))
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            grid_results[tuple(params.items())].append(test_accuracy)

    for params, results in grid_results.items():
        logging.info(f"Guidance Pipeline - Params: {dict(params)}")
        logging.info(f"Guidance Pipeline - Results: {results}")

    best_params = max(grid_results, key=lambda x: np.mean(grid_results[x]))
    logging.info(f"Guidance Pipeline - Last Year {last_year} - Best Params: {dict(best_params)}")
    logging.info(f"Guidance Pipeline - Last Year {last_year} - Best Score: {np.mean(grid_results[best_params])}")
    best_params = dict(best_params)

    best_model = SVC(**best_params)
    best_model.fit(train_embeddings, train_labels)
    train_predictions = best_model.predict(nonaugmented_train_embeddings)
    test_predictions = best_model.predict(nonaugmented_test_embeddings)

    # train_scores = best_model.predict_proba(nonaugmented_train_embeddings)[:, 1]
    # test_scores = best_model.predict_proba(nonaugmented_test_embeddings)[:, 1]
    # val_scores = best_model.predict_proba(nonaugmented_val_embeddings)[:, 1]
    train_scores = best_model.decision_function(nonaugmented_train_embeddings)
    test_scores = best_model.decision_function(nonaugmented_test_embeddings)

    train_accuracy = accuracy_score(nonaugmented_train_labels, train_predictions)
    test_accuracy = accuracy_score(nonaugmented_test_labels, test_predictions)

    print(f"Guidance Pipeline - Last Year {last_year} - Train Accuracy: {train_accuracy:.2f}")
    print(f"Guidance Pipeline - Last Year {last_year} - Test Accuracy: {test_accuracy:.2f}")
    print(f"Guidance Pipeline - Last Year {last_year} - Test Confusion Matrix: {confusion_matrix(nonaugmented_test_labels, test_predictions)}")
    return train_scores, test_scores
    

if __name__ == "__main__":
    guidance_pipeline()