import re
# import pickle as pkl
import pandas as pd
import scipy.spatial
import silence_tensorflow.auto
import streamlit as st

silence_tensorflow.auto
from sentence_transformers import SentenceTransformer, util
from summarizer import Summarizer, TransformerSummarizer
from tqdm import tqdm


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str


def main():
    bert_model = Summarizer

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("hotelReviewsInLondon.csv")
    df['hotelName'].drop_duplicates()
    df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(
        name='all_review')

    df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

    df_combined['all_review'] = df_combined['all_review'].apply(lambda x: lower_case(x))
    df = df_combined
    df_sentences = df_combined.set_index("all_review")
    print(df_sentences.head())
    df_sentences = df_sentences["hotelName"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    print(len(df_sentences_list))
    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    corpus = df_sentences_list
    corpus_embeddings = embedder.encode(corpus)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    paraphrases = util.paraphrase_mining(model, corpus)
    # with open("corpus_embeddings.pkl", "wb") as file2:
    #     pkl.dump(corpus_embeddings, file2)
    st.header('Top 3 Hotels Based on Search')
    queries = st.text_input('Enter your query here:')
    query_embeddings = embedder.encode(queries)

    if st.button('Search'):
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
                print("Score:   ", "(Score: %.4f)" % (1 - distance), "\n")
                print("Paragraph:   ", corpus[idx].strip(), "\n")
                # bert_summary = ''.join(bert_model(corpus[idx].strip()))
                # print(bert_summary)
                row_dict = df.loc[df['all_review'] == corpus[idx]]
                print("paper_id:  ", row_dict['hotelName'], "\n")
                print("-------------------------------------------")


if __name__ == '__main__':
    main()
