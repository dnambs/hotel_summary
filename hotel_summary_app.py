import re
import pickle as pkl
import pandas as pd
import scipy.spatial

import streamlit as st
from sentence_transformers import SentenceTransformer, util
#from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from tqdm import tqdm




def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

def summarized_review(data):
    print("Data",data)
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    result = model(data, num_sentences=3)
    return result

def main():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("hotelReviewsInLondon.csv")

    df['hotelName'].drop_duplicates()
    df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(
        name='all_review')
    # = df_combined.head(2).copy()
    df_summary = df_combined.copy()
    df_summary['summary'] = df_combined['all_review'].apply(summarized_review)
    df_combined['summary'] = df_summary['summary']

    df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

    df_combined['all_review'] = df_combined['all_review'].apply(lambda x: lower_case(x))
    df = df_combined.copy()
    df_sentences = df_combined.set_index("all_review")

    df_sentences = df_sentences["hotelName"].to_dict()
    df_sentences_list = list(df_sentences.keys())

    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    corpus = df_sentences_list
    corpus_embeddings = embedder.encode(corpus)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paraphrases = util.paraphrase_mining(model, corpus)
    st.header('Top 3 Hotels Based on Search')

    queries = ['A hotel near Big Ben']
    query_embeddings = embedder.encode(queries)

    with open("corpus.pkl", "wb") as file1:
        pkl.dump(corpus, file1)

    with open("corpus_embeddings.pkl", "wb") as file2:
        pkl.dump(corpus_embeddings, file2)

    with open("df.pkl", "wb") as file3:
        pkl.dump(df, file3)

    closest_n = 3
    print("\nTop 3 most similar sentences in corpus:")
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n=========================================================")
        print("==========================Query==============================")
        print("===", query, "=====")
        print("=========================================================")

        for idx, distance in results[0:closest_n]:
            st.write("Score:   ", "(Score: %.4f)" % (1 - distance), "\n")
            st.write("Bert Summary: ", df['summary'][idx])
            st.write(df['summary'][idx])
            row_dict = df.loc[df['all_review'] == corpus[idx]]
            st.write("paper_id:  ", row_dict['hotelName'], "\n")
            st.write("-------------------------------------------")


if __name__ == '__main__':
    main()
