import json
import pandas as pd
import numpy as np
import logging
import os
import random
import re
import asyncio
from tqdm.asyncio import tqdm 

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

openai_key = "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"

embeddings_engine = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    openai_api_key = openai_key
)


reporters_prompt = """
    You are given a FED press conference. Your task is to extract all the reporters that asked questions to the chairman during this conference.

    Please output a names separated by the ;

    Press Conference:
    {press_conference}
"""

mgi_openai_key = "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"
chat = ChatOpenAI(api_key=mgi_openai_key, model="gpt-4o", temperature=0.2)

def get_reporters(press_conference):
    reporters = chat.invoke(reporters_prompt.format(press_conference=press_conference)).content
    reporters = reporters.split(';')
    reporters = [reporter.strip() for reporter in reporters]
    return reporters

async def async_get_reporters(press_conference):
    reporters = await chat.ainvoke(reporters_prompt.format(press_conference=press_conference))
    reporters = reporters.content
    reporters = reporters.split(';')
    reporters = [reporter.strip() for reporter in reporters]
    return reporters

async def cut_introduction(text: str):
    try:
        reporters = await async_get_reporters(text)
        reporter_indices = []
        for reporter in reporters:
            reporter_regex = re.compile(re.escape(reporter), re.IGNORECASE)
            match = reporter_regex.search(text)
            if match:
                reporter_indices.append(match.start())

        min_index = min(reporter_indices)
        introduction = text[:min_index]
    except BaseException as exc:
        introduction = text
    
    return introduction

async def get_introductions_async(texts):
    return await tqdm.gather(*[cut_introduction(text) for text in texts])

def get_introductions(qa_data: pd.DataFrame):
    texts = qa_data['text'].tolist()
    introductions = asyncio.run(get_introductions_async(texts))
    qa_data['introduction'] = introductions
    return qa_data

def generate_pairs_ranking(qa_df, initial_pairs_ranking):
    result_pairs_ranking = {}

    for pair, value in initial_pairs_ranking.items():
        try:
            if value == 1:
                higher_text = pair[0]
                lower_text = pair[1]
            else:
                lower_text = pair[0]
                higher_text = pair[1]
                
            higher_text_date = qa_df.loc[qa_df['text'] == higher_text]['date'].values[0]
            lower_text_date = qa_df.loc[qa_df['text'] == lower_text]['date'].values[0]

            higher_text_introduction = qa_df.loc[(qa_df['date'] == higher_text_date) & (qa_df['shift'].isna())]['introduction'].values[0]
            lower_text_introduction = qa_df.loc[(qa_df['date'] == lower_text_date) & (qa_df['shift'].isna())]['introduction'].values[0]

            higher_text_hawkish = qa_df.loc[(qa_df['date'] == higher_text_date) & (qa_df['shift'] == 'hawkish')]['introduction'].values[0]
            higher_text_dovish = qa_df.loc[(qa_df['date'] == higher_text_date) & (qa_df['shift'] == 'dovish')]['introduction'].values[0]

            lower_text_hawkish = qa_df.loc[(qa_df['date'] == lower_text_date) & (qa_df['shift'] == 'hawkish')]['introduction'].values[0]
            lower_text_dovish = qa_df.loc[(qa_df['date'] == lower_text_date) & (qa_df['shift'] == 'dovish')]['introduction'].values[0]

            coinflip = random.random() > 0.5 

            if coinflip:
                #Infered Pairs for higher and lower texts
                result_pairs_ranking[(higher_text_hawkish, lower_text_introduction)] =  abs(value)
                result_pairs_ranking[(higher_text_introduction, lower_text_dovish)] =  abs(value)
                result_pairs_ranking[(higher_text_hawkish, lower_text_dovish)] =  1

            else:
                result_pairs_ranking[(lower_text_introduction, higher_text_hawkish)] =  -1 * abs(value)
                result_pairs_ranking[(lower_text_dovish, higher_text_introduction)] =  -1 * abs(value)
                result_pairs_ranking[(lower_text_dovish, higher_text_hawkish)] =  -1

            #Original Pairs from the triplets 
            coinflip = random.random() > 0.5 
            if coinflip:
                result_pairs_ranking[(higher_text_hawkish, higher_text_introduction)] =  0.5
                result_pairs_ranking[(higher_text_introduction, higher_text_dovish)] =  0.5
                result_pairs_ranking[(higher_text_hawkish, higher_text_dovish)] =  1
            else:
                result_pairs_ranking[(higher_text_introduction, higher_text_hawkish)] =  -0.5
                result_pairs_ranking[(higher_text_dovish, higher_text_introduction)] =  -0.5
                result_pairs_ranking[(higher_text_dovish, higher_text_hawkish)] =  -1

            coinflip = random.random() > 0.5 
            if coinflip:
                result_pairs_ranking[(lower_text_hawkish, lower_text_introduction)] =  0.5
                result_pairs_ranking[(lower_text_introduction, lower_text_dovish)] =  0.5
                result_pairs_ranking[(lower_text_hawkish, lower_text_dovish)] =  1
            else:
                result_pairs_ranking[(lower_text_introduction, lower_text_hawkish)] =  -0.5
                result_pairs_ranking[(lower_text_dovish, lower_text_introduction)] =  -0.5
                result_pairs_ranking[(lower_text_dovish, lower_text_hawkish)] =  -1


        except BaseException as exc: #We dodge exceptions for blocks where some augmentations are missing
            pass 

    return result_pairs_ranking

def qa_pairs_to_introductions():
    df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled_with_introductions.csv")
    
    introductions_years_pairs_ranking = {}
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/years_pairs_ranking.json", "r") as f:
        years_pairs_ranking = json.load(f)
        for year, pairs in years_pairs_ranking.items():
            introductions_years_pairs_ranking[year] = {}
            for pair, value in pairs.items():
                pair = json.loads(pair)
                first = df.loc[df['text'] == pair[0]]['introduction'].values[0]
                second = df.loc[df['text'] == pair[1]]['introduction'].values[0]
                introductions_years_pairs_ranking[year][(first, second)] = value

                additional_pairs = generate_pairs_ranking(df, pairs) #generate introduction-introduction pairs but based on the original pairs. 
                introductions_years_pairs_ranking[year].update(additional_pairs)

    return introductions_years_pairs_ranking

def embed_pairs(yearly_pairs: list[tuple[str, str]]):
    texts = set()
    for year, pairs in yearly_pairs.items():
        for pair in pairs:
            texts.add(pair[0])
            texts.add(pair[1])

    embeddings = embeddings_engine.embed_documents(list(texts))
    text_to_embedding = dict(zip(texts, embeddings))

    return text_to_embedding



if __name__ == "__main__":
    # qa_data = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")
    # qa_data = get_introductions(qa_data)
    # qa_data.to_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled_with_introductions.csv", index=False)

    qa_intro_pairs = qa_pairs_to_introductions()
    embeds = embed_pairs(qa_intro_pairs)

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/introductions_embeddings.json", "w") as f:
        json.dump(embeds, f)

    qa_intro_pairs = {year: {json.dumps(pair): value for pair, value in pairs.items()} for year, pairs in qa_intro_pairs.items()}
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/introductions_years_pairs_ranking.json", "w") as f:
        json.dump(qa_intro_pairs, f)
