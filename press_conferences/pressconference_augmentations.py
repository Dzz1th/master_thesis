import pandas as pd
from pydantic import BaseModel, Field
import random
import asyncio
from tqdm.asyncio import tqdm   
from utils import TokenRateLimiter
import json 
import logging
from langchain_community.callbacks import get_openai_callback

import nest_asyncio
nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_costs.log'),
        logging.StreamHandler()
    ]
)

rate_limiter = TokenRateLimiter(150, 30000000)

mgi_openai_key = "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"
openai_key = "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"

chat = ChatOpenAI(api_key=mgi_openai_key, model="gpt-4o", temperature=0.2)

prompt = """
    You are an international expert in macroeconomics and central bank policy.
    You are given a 2 press conference transcript.
    Your task is to compare these transcripts on the scale of hawkish/dovish and determine which one is more hawkish/dovish.
    Please provide a detailed analysis and reasoning for your answer. Please focus on a language and the narrative of the press conference.

    Press conference transcript 1:
    {transcript_1}

    Press conference transcript 2:
    {transcript_2}

    Your answer:
"""

async def compare_transcripts(transcript_1: str, transcript_2: str) -> str:
    # Use the rate limiter to manage the request
    request = prompt.format(transcript_1=transcript_1, transcript_2=transcript_2)
    await rate_limiter.wait_for_token_availability(len(request))
    
    with get_openai_callback() as cb:
        async with rate_limiter.request_limiter:
            response = await chat.ainvoke([HumanMessage(content=request)])
        
        # Log token usage and cost
        logging.info(f"Tokens: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        logging.info(f"Cost: ${cb.total_cost:.4f}")
        
    return (transcript_1, transcript_2), response.content

def get_pairs(texts_1, texts_2):
    pairs = []
    for i in range(len(texts_1)):
        for j in range(len(texts_2)):
            pairs.append((texts_1[i], texts_2[j]))
    return pairs

def generate_pairs_dataset(df):
    neutral_df = df[df['shift'].isna()]

    neutral_neutral_pairs = get_pairs(neutral_df['text'].tolist(), neutral_df['text'].tolist())

    return neutral_neutral_pairs

async def generate_pairs():
    total_tokens = 0
    total_cost = 0
    
    qa_df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")


    train_df = qa_df[qa_df['date'] < '2020-01-01']
    test_df = qa_df[qa_df['date'] >= '2020-01-01']

    #We generate pairs only for neutral-neutral pairs.
    #Because we don't really need scoring for hawkish/dovish pairs, if we are certain in the neural labels
    #So for each pair where we know neutral-neutral pairs, we also know neutral-dovish or neutral-hawkish pair depending on the relation
    train_pairs = generate_pairs_dataset(train_df) 

    train_pairs = random.sample(train_pairs, 300)

    train_pairs_results = {}

    tasks = [compare_transcripts(train_pairs[i][0], train_pairs[i][1]) for i in range(len(train_pairs))]
    
    logging.info(f"Generating train pairs: from 2011 to 2020")
    # Use tqdm to monitor the progress of the tasks
    results = await tqdm.gather(*tasks, desc="Processing Transcripts", total=len(tasks))

    for result in results:
        train_pairs_results[json.dumps(result[0])] = result[1]

    # Log final statistics
    logging.info(f"Generation completed:")
    logging.info(f"Total pairs processed: {len(train_pairs)}")
    
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/train_pairs_analysis.json", "w") as f:
        json.dump(train_pairs_results, f)

    test_pairs = generate_pairs_dataset(test_df)
    test_pairs = random.sample(test_pairs, 300)

    test_pairs_results = {}

    logging.info(f"Generating test pairs: from 2020 to 2024")
    test_tasks = [compare_transcripts(test_pairs[i][0], test_pairs[i][1]) for i in range(len(test_pairs))]
    test_results = await tqdm.gather(*test_tasks, desc="Processing Transcripts", total=len(test_tasks))

    for result in test_results:
        test_pairs_results[json.dumps(result[0])] = result[1]

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/test_pairs_analysis.json", "w") as f:
        json.dump(test_pairs_results, f)

async def generate_pairs_yoy():
    qa_df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")

    years_results = {}
    qa_df['date'] = pd.to_datetime(qa_df['date'])
    years = qa_df['date'].dt.year.unique()
    for year in years:
        years_results[int(year)] = {}
        year_df = qa_df[qa_df['date'].dt.year == year]
        year_pairs = generate_pairs_dataset(year_df)
        tasks = [compare_transcripts(year_pairs[i][0], year_pairs[i][1]) for i in range(len(year_pairs))]
    
        logging.info(f"Generating train pairs for year {year}")
        # Use tqdm to monitor the progress of the tasks
        results = await tqdm.gather(*tasks, desc="Processing Transcripts", total=len(tasks))

        for result in results:
            years_results[int(year)][json.dumps(result[0])] = result[1]

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/years_pairs_analysis.json", "w") as f:
        json.dump(years_results, f)




class FilterResponse(BaseModel):
    score: float = Field(description="Score indicating how much more hawkish/dovish the first press conference is compared to the second one")


filter_prompt = """
    You are an international expert in macroeconomics and central bank policy.
    You are given an analysis regarding the difference between hawkish/dovish sentiment and language of 2 press conferences. 
    Your task is to read an alaysis and rank which press conference was more hawkish. If press conference 1 is more hawkish -> 1 > 2, if press conference 1 is more dovish -> 1 < 2.
    You should also analyze how much more hawkish/dovish the first press conference is compared to the second one. 
    You must return 1 if the first press conference is significantly more hawkish, 0.5 if first press conference is slightly more hawkish than the second one, -0.5 if first press conference is more dovish than the second one, and -1 if the first press conference is significantly more dovish. If 2 press conferences do not differ in sentiment and language return 0.
    Output object with field score, which is a number from {{-1, -0.5, 0, 0.5, 1}}.

    Analysis:
    {analysis}

    Your answer:
"""

async def get_score(analysis: str):
    request = filter_prompt.format(analysis=analysis)
    structured_chat = chat.with_structured_output(FilterResponse)
    await rate_limiter.wait_for_token_availability(len(request))
    
    with get_openai_callback() as cb:
        async with rate_limiter.request_limiter:
            result = await structured_chat.ainvoke(request)
    
    return result.score


async def filter_pairs():
    total_tokens = 0
    total_cost = 0
    
    train_pairs_results = json.load(open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/train_pairs_analysis.json"))
    train_pairs_list = [(json.loads(key), value) for key, value in train_pairs_results.items()]
    train_analysis_list = [train_pairs_list[i][1] for i in range(len(train_pairs_list))]
    tasks = [get_score(train_analysis_list[i]) for i in range(len(train_analysis_list))]

    results = await tqdm.gather(*tasks, desc="Processing Transcripts", total=len(tasks))

    for i in range(len(results)):
        train_pairs_results[json.dumps(train_pairs_list[i][0])] = results[i]

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/train_pairs_ranking.json", "w") as f:
        json.dump(train_pairs_results, f)

    test_pairs_results = json.load(open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/test_pairs_analysis.json"))
    test_pairs_list = [(json.loads(key), value) for key, value in test_pairs_results.items()]
    test_analysis_list = [test_pairs_list[i][1] for i in range(len(test_pairs_list))]
    test_tasks = [get_score(test_analysis_list[i]) for i in range(len(test_analysis_list))]
    test_results = await tqdm.gather(*test_tasks, desc="Processing Transcripts", total=len(test_tasks))

    for i in range(len(test_results)):
        test_pairs_results[json.dumps(test_pairs_list[i][0])] = test_results[i]

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/test_pairs_ranking.json", "w") as f:
        json.dump(test_pairs_results, f)

async def filter_pairs_yoy():
    years_results = json.load(open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/years_pairs_analysis.json"))

    for year in years_results:
        pairs_list = list(years_results[year].keys())
        tasks = [get_score(years_results[year][pairs_list[i]]) for i in range(len(pairs_list))]
        results = await tqdm.gather(*tasks, desc="Processing Transcripts", total=len(tasks))

        for i in range(len(results)):
            years_results[year][pairs_list[i]] = results[i]

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/years_pairs_ranking.json", "w") as f:
        json.dump(years_results, f)


summary_augmentation_prompt = """
    You are an expert in macroeconomics and central bank policy.
    You are given a set of phrases from a FED press conference transcript.
    These phrases describe the stance of the FED in this press conference. 
    Your task is to generate a same set of phrases, that could describe the stance of the FED in this press conference if it was more {stance}.
    Make sure that the new set of phrases is coherent with the original set of phrases and reflect the change in the stance of the FED.
    You should return only a list of phrases.

    Original phrases:
    {phrases}

    Your answer:
"""


if __name__ == "__main__":
    asyncio.run(filter_pairs_yoy())