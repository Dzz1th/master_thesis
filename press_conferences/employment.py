import pandas as pd 
import re
import typing as t
from tqdm import tqdm
from utils import TokenRateLimiter
import logging 
import numpy as np
import tiktoken
import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from cache import set_cache

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

set_cache()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('employment_pipeline.log'),
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

employment_classification_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a set of statements from a FED press conference transcript.
    Your task is to analyze this set of phrases and determine what was the main focus of the FED in this press conference.
    If FED was more focused on supporting employment, return "employment". If FED was more focused on fighting inflation, return "inflation".

    Please output your analysis and the result. Analysis should be in 1-2 sentences. Finish your answer with "Result: " and put result into ## .

    Press Conference main statements:
    {statements}

    Your answer:
"""

async def classify_employment(statements):
    prompt = employment_classification_prompt.format(statements=statements)
    tokens_needed = tiktoken.encoding_for_model("gpt-4o").encode(prompt)
    await rate_limiter.wait_for_token_availability(len(tokens_needed))
    response = await chat.ainvoke(prompt)
    return response.content

async def classify_employment_statements(statements):
    tasks = [classify_employment(statements[i]) for i in range(len(statements))]
    results = await tqdm_async.gather(*tasks, desc="Processing Transcripts", total=len(tasks))
    
    parsed_results = []
    for response in results:    
        result = re.search(r"Result: ##(.*)##", response)
        result = result.group(1).strip()
        parsed_results.append(result)

    return parsed_results


employment_augmentation_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a subset of statements from a FED press conference transcript that are helping to understand whether the FED is more focused on supporting employment or fighting inflation.
    Your task is to generate a similar set of phrases that {task}

    You should use the provided phrases as a reference for a style and order. Think about which data and events would accompany the statements that {task}.
    You should return only a list of phrases.

    Original phrases:
    {phrases}

    Your answer:
"""

async def augment_employment(statements: str, task: str) -> str:
    prompt = employment_augmentation_prompt.format(phrases=statements)
    tokens_needed = tiktoken.encoding_for_model("gpt-4o").encode(prompt)
    await rate_limiter.wait_for_token_availability(len(tokens_needed))
    response = await chat.ainvoke(prompt)
    return response.content

async def augment_employment_statements(statements: t.List[str], classes: t.List[str]) -> str:
    inflation_task = "focus on rising inflation and the need to fight it"
    employment_task = "focus on supporting employment and growth"
    augmentation_tasks = [inflation_task if class_ == 'inflation' else employment_task for class_ in classes]
    tasks = [augment_employment(statements[i], augmentation_tasks[i]) for i in range(len(statements))]
    results = await tqdm_async.gather(*tasks, desc="Processing Transcripts", total=len(tasks))
    return results


def employment_pipeline(data, last_year: int = 2019, base_year: int = 2018) -> t.List[str]:
    """
        Employment pipeline.

        Train model using CV on train set from 2011 to last year.
        Using CV, derive best parameters.
        After that, train model with best parameters again using CV and form predictions for each fold.
        Obtain fold predictions and put them as train data for the next stage.
        Also return predictions for test set.
    """

    classes = asyncio.run(classify_employment_statements(data['employment_summary'].tolist()))
    data['employment_class'] = classes
    texts = data['employment_summary'].tolist()
    # augmented_employment = asyncio.run(augment_employment_statements(texts, classes))
    # data['employment_augmentations'] = augmented_employment
    #augmentation_classes = asyncio.run(tqdm.gather(*[classify_employment_statements(text) for text in augmented_employment]))
    #data['employment_augmentation_class'] = augmentation_classes

    train_data = data[data['date'] < f'{last_year}-01-01']
    test_data = data[(data['date'] >= f'{last_year}-01-01') & (data['date'] < f'{last_year+1}-01-01')]

    #train_groups = list(range(len(train_data))) + list(range(len(train_data)))
    train_texts = train_data['employment_summary'].tolist()

    train_labels = train_data['employment_class'].tolist()

    chairman_groups = train_data['chairman'].tolist()

    train_labels = np.array([0 if label == "employment" else 1 for label in train_labels])

    train_embeddings = np.array(embeddings_engine.embed_documents(train_texts))
    
    nonaugmented_train_embeddings = np.array(embeddings_engine.embed_documents(train_data['employment_summary'].tolist()))
    nonaugmented_test_embeddings = np.array(embeddings_engine.embed_documents(test_data['employment_summary'].tolist()))

    nonaugmented_train_labels = np.array([0 if label == "employment" else 1 for label in train_data['employment_class'].tolist()])
    nonaugmented_test_labels = np.array([0 if label == "employment" else 1 for label in test_data['employment_class'].tolist()])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = ParameterGrid({'C': [0.1, 1, 10, 100]})
    grid_results = {tuple(params.items()): [] for params in grid}

    #Stratify on chairman labels and ensure that augmentations of the train do not get mixed into the validation set
    for train_index, test_index in skf.split(train_embeddings, chairman_groups):
        X_train, X_test = train_embeddings[train_index], train_embeddings[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        for params in grid:
            model = SVC(**dict(params))
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            grid_results[tuple(params.items())].append(test_accuracy)

    for params, results in grid_results.items():
        logging.info(f"Employment Pipeline - Last Year {last_year} - Params: {dict(params)}")
        logging.info(f"Employment Pipeline - Last Year {last_year} - Results: {results}")

    best_params = max(grid_results, key=lambda x: np.mean(grid_results[x]))
    logging.info(f"Employment Pipeline - Last Year {last_year} - Best Params: {dict(best_params)}")
    logging.info(f"Employment Pipeline - Last Year {last_year} - Best Score: {np.mean(grid_results[best_params])}")
    best_params = dict(best_params)

    best_model = SVC(**best_params)
    best_model.fit(train_embeddings, train_labels)

    train_predictions = best_model.predict(nonaugmented_train_embeddings)
    test_predictions = best_model.predict(nonaugmented_test_embeddings)

    # train_scores = best_model.predict_proba(nonaugmented_train_embeddings)[:, 1]
    # val_scores = best_model.predict_proba(nonaugmented_val_embeddings)[:, 1]
    # test_scores = best_model.predict_proba(nonaugmented_test_embeddings)[:, 1]
    train_scores = best_model.decision_function(nonaugmented_train_embeddings)
    test_scores = best_model.decision_function(nonaugmented_test_embeddings)

    train_accuracy = accuracy_score(nonaugmented_train_labels, train_predictions)
    test_accuracy = accuracy_score(nonaugmented_test_labels, test_predictions)
    logging.info(f"Employment Pipeline - Last Year {last_year} - Train Accuracy: {train_accuracy:.2f}")
    logging.info(f"Employment Pipeline - Last Year {last_year} - Test Accuracy: {test_accuracy:.2f}")
    logging.info(f"Employment Pipeline - Last Year {last_year} - Test Confusion Matrix: {confusion_matrix(nonaugmented_test_labels, test_predictions)}")

    return train_scores, test_scores


if __name__ == "__main__":
    employment_pipeline(last_year=2019)





