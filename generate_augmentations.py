import pandas as pd 
import json 
import asyncio
import typing as t
import re
import numpy as np
from itertools import chain

from langchain_openai import ChatOpenAI
from tqdm import tqdm
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from prompts import PROMPTS

from concurrent import futures 

openai_api_key = "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"
chat = ChatOpenAI(temperature=0.5, model="gpt-4o", openai_api_key=openai_api_key)

import logging
import time

async def process_inputs(input_list, user_tier, chain):
    try:
        max_retries = 6
        delay_increment = 60

        # Optimized batch size calculation
        batch_size = min(30 if user_tier < 4 else 80, len(input_list))
        logging.info(f"Batch Size: {batch_size}")

        for i in range(0, len(input_list), batch_size):
            batch = input_list[i : i + batch_size]
            logging.info(f"Node Number: {node_number}, Batch Index: {i}")

            retries = 0
            while retries <= max_retries:
                try:
                    result = await chain.apply(batch)
                    logging.info(f"Chain Result: {result} for Input Batch: {batch}")
                    break  # Exit the retry loop once successful

                except RateLimitError as rate_limit_error:
                    delay = (retries + 1) * delay_increment
                    logging.warning(f"{rate_limit_error}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1

                    if retries > max_retries:
                        logging.error(f"Max retries reached for batch starting at index {i}. Skipping to next batch.")
                        break

                except KeyError as key_missing_error:
                    logging.error(f"Key missing: {key_missing_error}")
                    break
                except Exception as e:
                    logging.error(f"Error in process_inputs: {e}")
                    break

        logging.info(f"Final Results: {results}")
    except Exception as e:
        logging.error(f"An error occurred in process_inputs: {e}")
        results = []
    return results

def generate_with_plan(text, prompts):
    plan_prompt, report_prompt = prompts[0], prompts[1]
    strategies = chat.invoke(plan_prompt.format(minutes=text)).content
    try:
        strategies = strategies[strategies.index("##Strategies") + len("##Strategies"):].strip()
    except BaseException as exc:
        print(exc)
        
    system_message = SystemMessage("You are a highly regarded expert in macroeconomics and central bank policy.")
    initial_message = HumanMessage(content=report_prompt.format(minutes=text, strategies=strategies))
    report = ''
    
    messages = [system_message, initial_message]
    new_report = chat.invoke(messages).content
    ai_message = AIMessage(content=new_report)
    messages.append(ai_message)
    report += new_report
    
    # for i in range(1):
    #     message = HumanMessage(
    #         "Continue generating your report"
    #     )
    #     messages.append(message)
    #     new_report = chat.invoke(messages).content
    #     ai_message = AIMessage(content=new_report)
    #     messages.append(ai_message)
    #     report += new_report
        
    return report

def generate_scores_(text: str, score_prompt: str, axes: t.List):
    """
        Generate scores for a single text using the scoring prompt

        Args:
            text: str - FED's minutes
            score_prompt: str - scoring prompt
            axes: t.List[str] - list of axes along which we score, with their description and scoring rule.
        
        Returns:
            scores: t.Dict[str, Dict] of form 
                {
                    'axis': {
                        'score': from 0 to 10,
                        'reasoning': reasoning for the score
                    }
                }
    """
    scores = chat.invoke(score_prompt.format(text=text, axes=axes)).content
    match = re.search(r'```json\n(.*?)\n```', scores, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error decoding for text {text[:100]} with error: {e}")
            return {}
    else:
        print(f"Didn't find a json match for score object for text {text[:100]}")
        return {}
    
    
def generate_augmentations(texts: t.List[str], 
                           positive_dir_prompts: str,
                           negative_dir_prompts: str,
                           batch_size: 3):
    """
        Generate FED's minutes augmentation stretching them towards narrative lines.

        Args:
            texts: t.List[str] - original FED's minutes
            positive_dir_prompts: t.List[str] - prompts for generate_with_plan for positive directions
            negative_dir_prompts: t.List[str] - prompts for generate_with_plan for negative directions

        By positive/negative directions we mean the initial direction for strech (for example explicit guidance) and opposite for that direction

        Returns:
            results: t.Dict[str, Dict] of form 
                {
                    'text': str - original text
                    'positive': str - augmented text
                    'negative': str - augmented text
                }
    """
    results = []
    for text in tqdm(texts):
        with futures.ThreadPoolExecutor() as executor:
            explicit = []
            implicit = []
            for _ in range(batch_size):
                explicit.append(executor.submit(generate_with_plan, text, positive_dir_prompts))
                implicit.append(executor.submit(generate_with_plan, text, negative_dir_prompts))

            futures_ = explicit + implicit
            futures_ = [future.result() for future in futures.as_completed(futures_)]
            explicit = futures_[:batch_size]
            implicit = futures_[batch_size:]

            obj = {
                'text': text,
                'positive': explicit,
                'negative': implicit
            }
            results.append(obj)

    return results
    
def generate_scores(texts: t.List[str], 
                    score_prompt: str,
                    axes: t.List[str],
                    batch_size: int = 10):
    """
        Generate scores for FED's minutes in a multi-threaded manner
    """
    texts_with_scores = []
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batches):
        with futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(generate_scores_, text, score_prompt, axes) for text in batch]
            scores = [job.result() for job in futures.as_completed(jobs)]
            texts_with_scores += [{'text': batch[i], 'scores': score} for i, score in enumerate(scores)]

    return texts_with_scores

def generate_statement_augmentation(text, prompt):
    """
        This function is simple because statements are small and can be augmented in a very straightforward way
    """
    statement = chat.invoke(prompt.format(text=text)).content
    return statement 

async def generate_statements_augmentations(texts: t.List[str],
                                      augmentation_prompt: str, 
                                      axes: str,
                                      changes: t.List[str]):
    """
        Generate FED's minutes augmentation stretching them towards narrative lines.

        Args:
            texts: t.List[str] - original FED's minutes
            augmentation_prompts: t.List[str] - prompts for streching along specific axis

    """
    results = []
    prompts = []
    for text in texts:
        for change in changes:
            prompts.append(augmentation_prompt.format(text=text, axes=axes, change=change))
    
    results = await chat.abatch(prompts)
    results = [result.content for result in results]
    return results




if __name__ == "__main__":
    # df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/Fed_Scrape-2000-2024.csv")

    # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # df = df.drop_duplicates(subset=['Date'], keep='first')
    # df = df.reset_index(drop=True)
    # df = df[['Date', 'Text']]

    # texts = df['Text'].to_list()
    # train_texts = texts
    # test_texts = texts[80:]
    
    # results = []

    # with open("augmentations_guidance.json", "r") as f:
    #     results = json.load(f)

    # texts = [
    #     [result['text']]+result['explicit'] + result['implicit']
    #      for result in results
    #      ]
    
    # texts = list(chain(*texts))
    statements = pd.read_csv("statements.csv")
    texts = statements['Statement'].to_list()

    prompt_obj = next(prompt for prompt in PROMPTS if prompt['name'] == 'generate_statements')
    axes = prompt_obj['basic_axes']
    axes = '\n'.join(axes)
    changes = prompt_obj['basic_changes']
    generate_prompt = prompt_obj['prompt']

    augmented_statements = asyncio.run(generate_statements_augmentations(texts, generate_prompt, axes, changes))

    with open('augmented_statements.json', 'w') as f:
        json.dump(augmented_statements, f)


    # generate_explicit_guidance_prompt = next(prompt for prompt in PROMPTS if prompt['name'] == 'generate_explicit_guidance')
    # generate_implicit_guidance_prompt = next(prompt for prompt in PROMPTS if prompt['name'] == 'generate_implicit_guidance')
    # generate_explicit_minutes_prompt = next(prompt for prompt in PROMPTS if prompt['name'] == 'generate_explicit_minutes')
    # generate_implicit_minutes_prompt = next(prompt for prompt in PROMPTS if prompt['name'] == 'generate_implicit_minutes')

    # positive_prompts = (generate_explicit_guidance_prompt, generate_explicit_minutes_prompt)
    # negative_prompts = (generate_implicit_guidance_prompt, generate_implicit_minutes_prompt)

    # results = generate_augmentations(train_texts, positive_prompts, negative_prompts)
    
    # with open('augmentations_guidance.json', 'w') as f:
    #     json.dump(results, f)