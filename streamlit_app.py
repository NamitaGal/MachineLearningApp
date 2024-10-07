import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('this is an app bulids a machine learning model')

df = pd.read_csv("https://www.kaggle.com/code/rv1922/penguins-analysis-eda/input?scriptVersionId=191134290")
df
