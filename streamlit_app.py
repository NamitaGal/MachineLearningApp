import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('this is an app bulids a machine learning model')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv")
df
