import re
import math
import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Hashable


def divide_hash_in_blocks(hash: List[int], block_size: int = 32) -> List[int]:
    if len(hash) % block_size != 0:
        raise ValueError("The length of the hash must be a multiple of n_bits.")

    return [int("".join(map(str, hash[i:i + block_size])), 2) for i in range(0, len(hash), block_size)]


def simHash(terms_tf_idf: DataFrame, terms_rv: Dict[Hashable, np.ndarray], m: int = 64) -> List[int]:
    # Error handling
    if not isinstance(terms_tf_idf, pd.DataFrame):
        raise ValueError("terms_tf_idf must be a pandas DataFrame")
    if not isinstance(terms_rv, dict):
        raise ValueError("terms_rv must be a dictionary")
    if not isinstance(m, int):
        raise ValueError("m must be an integer")

    sim_hash = np.zeros(m)

    for term, tf_idf in terms_tf_idf.iterrows():
        if tf_idf["TF-IDF"] == 0:
            continue

        term_hash = terms_rv[term]
        sim_hash += term_hash * tf_idf["TF-IDF"]

    return (sim_hash >= 0).astype(int).tolist()


def cosine_similarity(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError("The hashes must have the same length.")

    hamming_distance = sum([1 for i, j in zip(hash1, hash2) if i != j])
    return 1 - hamming_distance / len(hash1)


def find_similar_documents(sim_hashes: List[List[int]], threshold: float = 0.9):
    similar_docs = []
    for i in tqdm.tqdm(range(len(sim_hashes))):
        for j in range(i + 1, len(sim_hashes)):
            similarity = cosine_similarity(sim_hashes[i], sim_hashes[j])
            if similarity >= threshold:
                similar_docs.append((i, j, similarity))

    return similar_docs


def find_similar_documents_optimized(sim_hashes_blocked: List[List[int]], block_size: int = 32, threshold: int = 0.9):
    # group the simHashes that shares at least one block
    pass


def save_string_to_file(string, filename):
    with open(filename, 'a') as file:
        file.write(string)


def main():
    # load parquet file with the documents
    dataset = pd.read_parquet("data/mail_1.parquet")
    print(dataset.head(), "\n", dataset.shape, "\n")

    dataset = pd.read_csv("data/news.csv", header=0)
    print(dataset.head(), "\n", dataset.shape, "\n")

    # select only the summary column and rename it to text
    dataset = dataset[["summary"]]
    dataset.rename(columns={"summary": "text"}, inplace=True)

    # set the number of docs to compute the simHash
    n_docs = 1000

    # Set the number of bits for the simHash
    m = 64

    # get first n_docs documents
    dataset = dataset.head(n_docs)

    # reset the index
    dataset.reset_index(drop=True, inplace=True)

    # Compute TF-IDF for each document
    tf_idf_vect = TfidfVectorizer(min_df=0.005)
    tf_idf = tf_idf_vect.fit_transform(dataset['text'])
    print(tf_idf_vect.get_feature_names_out(), '\n')

    # add number of terms > 0 in the dataset and sort both the dataset and the tf_idf matrix
    dataset["n_terms"] = [len(tf_idf[i].data) for i in range(n_docs)]
    dataset = dataset.sort_values(by="n_terms", ascending=True)
    tf_idf = tf_idf[dataset.index]

    # create a random vector {-1, +1}^m foreach term
    terms_rv = {term: np.random.choice([-1, 1], m) for term in tf_idf_vect.get_feature_names_out()}

    # Compute simHash for each document
    sim_hashes = []
    sim_hashes_blocked = []
    for i in tqdm.tqdm(range(n_docs)):
        # create a dataframe with the tf-idf values of terms in the document (tf-idf > 0)
        df = pd.DataFrame(tf_idf[i].T.todense(), index=tf_idf_vect.get_feature_names_out(), columns=["TF-IDF"])
        df = df.loc[df["TF-IDF"] > 0]
        sim_hashes.append(simHash(df, terms_rv, m))
        sim_hashes_blocked.append(divide_hash_in_blocks(simHash(df, terms_rv, m)))

    # Compute cosine similarity between the simHashes
    similar_docs = find_similar_documents(sim_hashes, threshold=0.9)

    print("len(similar_docs):", len(similar_docs))

    # save the text of similar documents in a file without blank spaces
    # empty file
    with open("similar_docs.txt", 'w') as file:
        file.write("")

    for i, j, similarity in similar_docs:
        # print(f"\nDocument {i} and Document {j} are similar with a similarity of {similarity}")
        save_string_to_file(f"Document {i} and Document {j} are similar with a similarity of {similarity}\n",
                            "similar_docs.txt")

        # get only words
        t1 = re.sub(r'[^\w ]', '', dataset['text'].loc[i])
        t2 = re.sub(r'[^\w ]', '', dataset['text'].loc[j])

        save_string_to_file(f"\n\nDocument {i}:\n{t1}\n", "similar_docs.txt")
        save_string_to_file(f"\n\nDocument {j}:\n{t2}\n\n\n", "similar_docs.txt")


if __name__ == "__main__":
    main()
