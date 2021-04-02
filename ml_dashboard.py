import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
import matplotlib.pyplot as plt
import sklearn
import pickle

# importing models
TFIDF_MODEL = './streamlit_data/tfidf'
NBC_MODEL = './streamlit_data/PAClassifierModel'

tfidf_v = load(TFIDF_MODEL)
classifier = load(NBC_MODEL)

# Setting up the page
st.title('Classifying insincere content in Social Media Posts')
st.write('Evaluate your questions against Quora Model!')
text_input  = st.text_input('Check your Quora Questions', value='', type='default')

if st.button('Check the Question'):
    X = tfidf_v.transform([text_input]) 
    predictions_test = classifier.predict(X)
    if predictions_test[0]:
        st.error(text_input + ' - IS INSINCERE')
    else:
        st.success(text_input + ' - IS SINCERE')
