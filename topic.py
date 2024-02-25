# topic_modeling_app.py

import streamlit as st
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample data
documents = [
    "Topic modeling is an interesting technique in natural language processing.",
    "It helps identify hidden topics in a collection of documents.",
    "Latent Dirichlet Allocation (LDA) is a popular algorithm for topic modeling.",
    "Streamlit is a great tool for building interactive web applications.",
]

# Preprocess the documents
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words

tokenized_docs = [preprocess_text(doc) for doc in documents]

# Create a dictionary and a corpus
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)

# Streamlit app
st.title("Topic Modeling App")

# User input
user_input = st.text_area("Enter text for topic modeling:")

if st.button("Generate Topics"):
    if user_input:
        # Preprocess user input
        input_tokens = preprocess_text(user_input)

        # Create a bag of words representation
        input_bow = dictionary.doc2bow(input_tokens)

        # Get the topic distribution
        topic_distribution = lda_model.get_document_topics(input_bow)

        # Display topics and probabilities
        st.write("Topic Distribution:")
        for topic, prob in topic_distribution:
            st.write(f"Topic {topic + 1}: Probability {prob:.4f}")
