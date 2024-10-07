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
  
with st.expander("visualization"):
  st.scatter_chart(data=df, x="bill_length_mm",y="body_mass_g",color="species")
# "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
st.sidebar.header("Input features")
island = st.sidebar.selectbox("Island",("Torgersen","Dream","Biscoe"))
gender = st.sidebar.selectbox("Gender",("female","male"))
bill_length = st.sidebar.slider("bill length(mm)",32.1,59.6,43.9)
bill_depth = st.sidebar.slider("bill depth(mm)",13.1,23.5,17.2)
flipper_length = st.sidebar.slider("flipper length(mm)",172.0,231.0,201.0)
body_mass = st.sidebar.slider("body mass(g)",2700.0,6300.0,4280.0)
