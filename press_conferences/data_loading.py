import json
import pandas as pd
import numpy as np
import logging
import os
import random

from sklearn.model_selection import train_test_split


def get_label_distribution(labels):
    labels_ = np.array(labels)
    return {
        "1's": labels_[labels_ == 1].shape[0] / labels_.shape[0],
        "0.5's": labels_[labels_ == 0.5].shape[0] / labels_.shape[0],
        "0's": labels_[labels_ == 0].shape[0] / labels_.shape[0],
        "-1's": labels_[labels_ == -1].shape[0] / labels_.shape[0],
        "-0.5's": labels_[labels_ == -0.5].shape[0] / labels_.shape[0]
    }

def load_pairs_data(test_size: float = 0.25, random_seed = None):
    """
        Load ranked pairs of texts, find unique texts, split them into train and test and then find all the pairs that consists ONLY of those texts.
        This approach fully avoids the leakage of data, but it cause the problem with inconsistent sample size and it is hard to stratify.

        Possible solution - undersample data to stratify. 
        
        Returns pairs of the embeddings and their respective rankings.
    """
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 
    
    pairs_ranking = data['pairs_ranking']

    pairs_ranking = {tuple(json.loads(key)): value for key, value in pairs_ranking.items()}

    train_idxs, test_idxs = train_test_split(list(text_idxs.values()), test_size=test_size, random_state=random_seed)

    train_pairs = {key: value for key, value in pairs_ranking.items() if key[0] in train_idxs and key[1] in train_idxs}
    test_pairs = {key: value for key, value in pairs_ranking.items() if key[0] in test_idxs and key[1] in test_idxs}

    logging.info(f"Train pairs: {len(train_pairs)}")
    logging.info(f"Test pairs: {len(test_pairs)}")

    train_embeddings = [(embeddings[key[0]], embeddings[key[1]]) for key in train_pairs.keys()]
    test_embeddings = [(embeddings[key[0]], embeddings[key[1]]) for key in test_pairs.keys()]
    train_labels = [value for value in train_pairs.values()]   
    test_labels = [value for value in test_pairs.values()]

    train_label_distribution = get_label_distribution(train_labels)
    test_label_distribution = get_label_distribution(test_labels)
    
    logging.info(f"Train label distribution: {train_label_distribution}")
    logging.info(f"Test label distribution: {test_label_distribution}")
    
    return train_embeddings, train_labels, test_embeddings, test_labels

def load_pairs_data_stratified(test_size: float = 0.25, random_seed = None):
    """
        This version of data loading read all the pairs and then splits them into train and test with stratification.

        Potential issue with this approach is that we will have same objects (but not same pairs) in train and test
            (altough we can have x1 < x2 < x3, and x1<x3 and x2<x3 in train, and x1<x3 in test).
        This could lead to the leakage in data and overfitting.
    """
    
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 
    
    pairs_ranking = data['pairs_ranking']
    pairs_ranking = {tuple(json.loads(key)): value for key, value in pairs_ranking.items()}

    pairs = list(pairs_ranking.keys())
    pairs = [(embeddings[pair[0]], embeddings[pair[1]]) for pair in pairs]
    labels = list(pairs_ranking.values())

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=test_size, stratify=labels, random_state=random_seed)

    train_label_distribution = get_label_distribution(train_labels)
    test_label_distribution = get_label_distribution(test_labels)

    logging.info(f"Train label distribution: {train_label_distribution}")
    logging.info(f"Test label distribution: {test_label_distribution}")

    return train_pairs, train_labels, test_pairs, test_labels


def load_pre_splited_pairs(validation_size: float = 0.2, random_seed = None):
    """
        Load pre-splited pairs from the files.
        Train sample is pairs generated from 2011 to 2020-01-01.
        Test sample is pairs generated from 2020-01-01 to 2021-01-01.

        We use simple train-val split on train data because its hard to create a validation set that will independently cover all the required narratives.
        So we hope that althought val score could be biased, dynamic of it will still allow us to find the best model configuration.
    """
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/train_pairs_ranking_linked.json", "r") as f:
        train_pairs_ranking = json.load(f)
        train_pairs_ranking = {tuple(json.loads(key)): value for key, value in train_pairs_ranking.items()}

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/test_pairs_ranking_linked.json", "r") as f:
        test_pairs_ranking = json.load(f)
        test_pairs_ranking = {tuple(json.loads(key)): value for key, value in test_pairs_ranking.items()}

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 

    text_to_embeddings = {text: embeddings[idx] for text, idx in text_idxs.items()}

    train_pairs, train_labels = [], []
    for pair, value in train_pairs_ranking.items():
        train_pairs.append((text_to_embeddings[pair[0]], text_to_embeddings[pair[1]]))
        train_labels.append(value)

    test_pairs, test_labels = [], []
    for pair, value in test_pairs_ranking.items():
        test_pairs.append((text_to_embeddings[pair[0]], text_to_embeddings[pair[1]]))
        test_labels.append(value)

    train_pairs, val_pairs, train_labels, val_labels = train_test_split(train_pairs, train_labels, test_size=validation_size, stratify=train_labels, random_state=random_seed)

    logging.info(f"Train pairs distribution: {get_label_distribution(train_labels)}")
    logging.info(f"Validation pairs distribution: {get_label_distribution(val_labels)}")
    logging.info(f"Test pairs distribution: {get_label_distribution(test_labels)}")

    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels

def load_time_grouped_pairs():
    qa_df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/years_pairs_ranking.json", "r") as f:
        years_pairs_ranking = json.load(f)
        for year, pairs in years_pairs_ranking.items():
            years_pairs_ranking[year] = {tuple(json.loads(key)): value for key, value in pairs.items()}
            years_pairs_ranking[year] = generate_pairs_ranking(qa_df, years_pairs_ranking[year])

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 

    text_to_embeddings = {text: embeddings[idx] for text, idx in text_idxs.items()}

    yearly_data = []
    for year in years_pairs_ranking:
        year_pairs, year_labels = [], []
        for pair, value in years_pairs_ranking[year].items():
            year_pairs.append((text_to_embeddings[pair[0]], text_to_embeddings[pair[1]]))
            year_labels.append(value)

        yearly_data.append((year_pairs, year_labels))

    return yearly_data

def load_introductions_years_pairs_ranking():
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/introductions_years_pairs_ranking.json", "r") as f:
        pairs_ranking = json.load(f)
        for year, pairs in pairs_ranking.items():
            pairs_ranking[year] = {tuple(json.loads(key)): value for key, value in pairs.items()}

    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/introductions_embeddings.json", "r") as f:
        embeddings = json.load(f)

    yearly_data = []
    for year, pairs in pairs_ranking.items():
        objects, labels = [], []
        for pair, value in pairs.items():
            objects.append((embeddings[pair[0]], embeddings[pair[1]]))
            labels.append(value)

        yearly_data.append((objects, labels))

    return yearly_data


def generate_pairs_ranking(qa_df, pairs_ranking):
    result_pairs_ranking = {}

    for pair, value in pairs_ranking.items():
        try:
            if value == 1:
                higher_text = pair[0]
                lower_text = pair[1]
            else:
                lower_text = pair[0]
                higher_text = pair[1]
                
            higher_text_date = qa_df.loc[qa_df['text'] == higher_text]['date'].values[0]
            lower_text_date = qa_df.loc[qa_df['text'] == lower_text]['date'].values[0]

            higher_text_hawkish = qa_df.loc[(qa_df['date'] == higher_text_date) & (qa_df['shift'] == 'hawkish')]['text'].values[0]
            higher_text_dovish = qa_df.loc[(qa_df['date'] == higher_text_date) & (qa_df['shift'] == 'dovish')]['text'].values[0]

            lower_text_hawkish = qa_df.loc[(qa_df['date'] == lower_text_date) & (qa_df['shift'] == 'hawkish')]['text'].values[0]
            lower_text_dovish = qa_df.loc[(qa_df['date'] == lower_text_date) & (qa_df['shift'] == 'dovish')]['text'].values[0]

            coinflip = random.random() > 0.5 

            if coinflip:
                #Infered Pairs for higher and lower texts
                result_pairs_ranking[(higher_text_hawkish, lower_text)] =  abs(value)
                result_pairs_ranking[(higher_text, lower_text_dovish)] =  abs(value)
                result_pairs_ranking[(higher_text_hawkish, lower_text_dovish)] =  1

            else:
                result_pairs_ranking[(lower_text, higher_text_hawkish)] =  -1 * abs(value)
                result_pairs_ranking[(lower_text_dovish, higher_text)] =  -1 * abs(value)
                result_pairs_ranking[(lower_text_dovish, higher_text_hawkish)] =  -1

            #Original Pairs from the triplets 
            coinflip = random.random() > 0.5 
            if coinflip:
                result_pairs_ranking[(higher_text_hawkish, higher_text)] =  0.5
                result_pairs_ranking[(higher_text, higher_text_dovish)] =  0.5
                result_pairs_ranking[(higher_text_hawkish, higher_text_dovish)] =  1
            else:
                result_pairs_ranking[(higher_text, higher_text_hawkish)] =  -0.5
                result_pairs_ranking[(higher_text_dovish, higher_text)] =  -0.5
                result_pairs_ranking[(higher_text_dovish, higher_text_hawkish)] =  -1

            coinflip = random.random() > 0.5 
            if coinflip:
                result_pairs_ranking[(lower_text_hawkish, lower_text)] =  0.5
                result_pairs_ranking[(lower_text, lower_text_dovish)] =  0.5
                result_pairs_ranking[(lower_text_hawkish, lower_text_dovish)] =  1
            else:
                result_pairs_ranking[(lower_text, lower_text_hawkish)] =  -0.5
                result_pairs_ranking[(lower_text_dovish, lower_text)] =  -0.5
                result_pairs_ranking[(lower_text_dovish, lower_text_hawkish)] =  -1


        except BaseException as exc: #We dodge exceptions for blocks where some augmentations are missing
            pass 

    return result_pairs_ranking

def load_triplets_data(date_from: str, date_to: str):
    qa_df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")
    qa_df = qa_df[(qa_df['date'] >= date_from) & (qa_df['date'] <= date_to)]

    qa_embeddings = np.load("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_embeddings.npy")
    qa_embeddings = qa_embeddings[qa_df.index]

    qa_df.reset_index(drop=True, inplace=True)

    def get_triplet(group):
        if (group['shift'] == 'hawkish').sum() == 1 and \
            (group['shift'] == 'dovish').sum() == 1 and \
            (group['shift'].isna()).sum() == 1:


            hawkish = group[group['shift'] == 'hawkish'].index.values[0]
            dovish = group[group['shift'] == 'dovish'].index.values[0]
            anchor = group[group['shift'].isna()].index.values[0]

            hawkish_embedding = qa_embeddings[hawkish]
            dovish_embedding = qa_embeddings[dovish]
            anchor_embedding = qa_embeddings[anchor]

            return (hawkish_embedding, dovish_embedding, anchor_embedding)
        else:
            return None

    triplets = qa_df.groupby("date").apply(get_triplet).dropna().tolist()
    return triplets 


def load_raw_qa_data():
    qa_df = pd.read_csv("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_data_labeled.csv")
    qa_embeddings = np.load("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/qa_embeddings.npy")
    return qa_df, qa_embeddings


if __name__ == "__main__":
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_pre_splited_pairs()


