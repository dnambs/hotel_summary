
import re
import pickle as pkl
import pandas as pd
import scipy.spatial
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
#from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from tqdm import tqdm

st.header("London Hotels")





def main():
    corpus = pd.read_pickle('corpus.pkl')
    corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
    df = pd.read_pickle('df.pkl')
    query = st.text_input('Enter your query here:')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # if st.button('Search'):
    #     closest_n = 3
    #     st.write("Top 3 most similar sentences in corpus:")
    #     for query, query_embedding in zip(query, query_embedding):
    #         #distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    #         distances = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    #         top_results = torch.topk(distances, k=closest_n)
    #
    #         #results = zip(range(len(distances)), distances)
    #         #results = sorted(results, key=lambda x: x[1])
    #
    #
    #         for score, idx in zip(top_results[0], top_results[1]):
    #
    #             st.write("Score:   ", score)
    #             st.write("Hotel: ", df['summary'][idx])
    #             st.write(df['summary'][idx])
    #             row_dict = df.loc[df['all_review'] == corpus[idx]]
    #             st.write("paper_id:  ", row_dict['hotelName'], "\n")
    #             st.write("-------------------------------------------")


if __name__ == '__main__':
    main()
