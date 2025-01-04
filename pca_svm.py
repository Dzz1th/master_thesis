import numpy as np 
import json
from sklearn.decomposition import PCA 

from sklearn.svm import SVC as SVM 
from sklearn.metrics import accuracy_score
from itertools import combinations, islice, chain
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def evaluate_permutation(perm, projected_train, projected_test, y_train, y_test):
    projected_train = projected_train[:, perm]
    projected_test = projected_test[:, perm]

    svm = SVM(kernel='linear')
    svm.fit(projected_train, y_train)
    predict = svm.predict(projected_test)
    
    return tuple(perm), accuracy_score(y_test, predict)

def process_permutation_chunk(chunk, projected_train_all, projected_test_all, y_train, y_test):
    results = {}
    for perm in chunk:
        perm_result = evaluate_permutation(perm, projected_train_all, projected_test_all, y_train, y_test)
        results[perm_result[0]] = perm_result[1]
    return results

def chunked_permutations(length, num_elems, chunk_size):
    iterator = combinations(list(range(num_elems)), length)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

def perform_pca_svm(train_data, test_data, y_train, y_test, n_components=50):
    results = {i: {} for i in range(1, 10)}

    with ProcessPoolExecutor() as executor:
        for i in tqdm(range(1, 5)):
            futures = []
            # for chunk in chunked_permutations(i, n_components, chunk_size):
            #     futures.append(executor.submit(process_permutation_chunk, chunk, projected_train_target, projected_test_target, y_train, y_test))
            chunks = list(combinations(list(range(n_components)), i))
            futures.append(executor.submit(process_permutation_chunk, chunks, train_data, test_data, y_train, y_test))
            for future in as_completed(futures):
                results[i].update(future.result())

            indexes = sorted(results[i].items(), key=lambda x: x[1], reverse=True)
            indexes = [index[0] for index in indexes]
            unique_elements = set()
            [unique_elements.add(elem) for tup in indexes for elem in tup if len(unique_elements) < 20]
            chunks = list(combinations(unique_elements, i+1))


    for key, value in results.items():
        results[key] = dict(sorted(value.items(), key=lambda x: x[1], reverse=True))

    results_ = {}
    for key, value in results.items():
        value_ = {}
        for k_, v_ in value.items():
            value_[str(k_)] = v_
        results_[key] = value_

    return results_

if __name__ == "__main__":
    # train_certainty_embed = np.load("./embeddings/train_certainty_embed.npy")
    # test_positive_certainty_embed = np.load("./embeddings/test_positive_certainty_embed.npy")
    # test_negative_certainty_embed = np.load("./embeddings/test_negative_certainty_embed.npy")
    # train_guidance_embed = np.load("./embeddings/train_guidance_embed.npy")
    # test_positive_guidance_embed = np.load("./embeddings/test_positive_guidance_embed.npy")
    # test_negative_guidance_embed = np.load("./embeddings/test_negative_guidance_embed.npy")
    topics = [
        'Guidance',
        'Certainty',
        'Direction',
        'Economic Outlook',
        'Commitment to Policy',
        'Inflation Targeting',
        'Market Reassurance'
    ]
    topic_results = {topic: {} for topic in topics}
    for topic in tqdm(topics):
        train_direction_embed = np.load(f"./embeddings/statements_train_neutral_{topic}_embeddings.npy")
        labels_direction_embed = np.load(f"./embeddings/statements_train_neutral_{topic}_labels.npy")
        positive = np.where(labels_direction_embed == 1)[0]
        negative = np.where(labels_direction_embed == -1)[0]
        neutral = np.where(labels_direction_embed == 0)[0]

        train_target_data = train_direction_embed
        mean_target_data = np.mean(train_target_data, axis=0)
        std_target_data = np.std(train_target_data, axis=0)
        train_target_data = (train_target_data - mean_target_data) / std_target_data

        pca = PCA(n_components=100)
        train_target_data = pca.fit_transform(train_target_data)

        train_pos_neg = np.concatenate([train_target_data[positive], train_target_data[negative]], axis=0)
        train_pos_neut = np.concatenate([train_target_data[positive], train_target_data[neutral]], axis=0)
        train_neg_neut = np.concatenate([train_target_data[negative], train_target_data[neutral]], axis=0)

        y_train_pos_neg = np.concatenate((np.ones(len(train_direction_embed[positive])),  -1*np.ones(len(train_direction_embed[negative]))))
        y_train_pos_neut = np.concatenate((np.ones(len(train_direction_embed[positive])),  -1*np.ones(len(train_direction_embed[neutral]))))
        y_train_neg_neut = np.concatenate((np.ones(len(train_direction_embed[negative])),  -1*np.ones(len(train_direction_embed[neutral]))))

        data = {
            'pos_neg': (train_pos_neg, y_train_pos_neg),
            'pos_neut': (train_pos_neut, y_train_pos_neut),
            'neg_neut': (train_neg_neut, y_train_neg_neut)
        }
        for key, (train_data, labels) in data.items():
            train_target_data = train_data
            test_target_data = train_target_data
            y_train = labels
            y_test = y_train

            svm = SVM(kernel='linear')
            svm.fit(train_target_data, y_train)
            predict = svm.predict(test_target_data)

            embed_normal = svm.coef_[0]
            embed_normal = embed_normal / np.linalg.norm(embed_normal)

            print('Embeddings Accuracy: ', accuracy_score(y_test, predict))

            results = perform_pca_svm(train_target_data, test_target_data, y_train, y_test, n_components=50)
            topic_results[topic][key] = results

            with open('statements_results.json', 'w') as f:
                json.dump(topic_results, f)