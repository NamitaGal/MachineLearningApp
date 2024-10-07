import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('this is an app bulids a machine learning model')

with st.expander("data"):
  st.write('**dataset**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df
  st.write('**x**')
  x = df.drop('species',axis=1)
  x
  st.write('**y**')
  y = df.species
  y
  
