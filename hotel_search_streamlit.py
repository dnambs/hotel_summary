
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
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.header("London Hotels")





def main():
    corpus = pd.read_pickle('corpus.pkl')
    corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
    df = pd.read_pickle('df.pkl')
    query = st.text_input('Enter your query here:')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    if st.button('Search'):
        closest_n = 3
        st.write("Top 3 most similar sentences in corpus:")
        #distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(distances, k=closest_n)

        #results = zip(range(len(distances)), distances)
        #results = sorted(results, key=lambda x: x[1])


        for score, idx in zip(top_results[0], top_results[1]):

            row_dict = df.loc[df['all_review'] == corpus[idx]]['hotelName'].values[0]
            st.markdown("<b>Hotel name:  </b>" + row_dict,unsafe_allow_html=True)
            summary_dict = df.loc[df['all_review'] == corpus[idx]]['summary'].values[0]
            st.markdown("<b>Summary: </b><p>"+ summary_dict +"</p>", unsafe_allow_html=True)

            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(corpus[idx])

            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.write("-------------------------------------------")



if __name__ == '__main__':
    main()
