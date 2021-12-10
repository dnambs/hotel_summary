import pandas as pd
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
from PIL import Image

import matplotlib.pyplot as plt

st.header("London Hotels")
image = Image.open('londonpic.jpg', )
st.image(image, caption='By Eva Dang')


def main():
    corpus = pd.read_pickle('corpus.pkl')
    corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
    df = pd.read_pickle('df.pkl')
    st.subheader('Enter the specifications for a hotel below:')
    query = st.text_input('')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    if st.button('Search'):
        closest_n = 3
        st.markdown("<b>Here are the top 3 hotels that match your search:</b>", unsafe_allow_html=True)
        # distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(distances, k=closest_n)

        # results = zip(range(len(distances)), distances)
        # results = sorted(results, key=lambda x: x[1])

        for score, idx in zip(top_results[0], top_results[1]):
            row_dict = df.loc[df['all_review'] == corpus[idx]]['hotelName'].values[0]
            st.markdown("<b>Hotel name:  </b>" + row_dict, unsafe_allow_html=True)
            summary_dict = df.loc[df['all_review'] == corpus[idx]]['summary'].values[0]
            st.markdown("<b>Summary: </b><p> " + summary_dict + "</p>", unsafe_allow_html=True)

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
