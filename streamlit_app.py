import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.title('Machine Learning App')

st.info('this is an app bulids a machine learning model')

with st.expander("data"):
  st.write('**dataset**')
  df = pd.read_csv("https://raw.githubusercontent.com/NamitaGal/MachineLearningApp/refs/heads/master/penguins_cleaned.csv")
  df
  st.write('**x**')
  x_raw = df.drop('species',axis=1)
  x_raw
  st.write('**y**')
  y_raw = df.species
  y_raw
  
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


data = {
  "island":island,
  "bill_length_mm":bill_length,
  "bill_depth_mm":bill_depth,
  "flipper_length_mm":flipper_length,
  "body_mass_g":body_mass,
  "sex":gender,
}
input_data = pd.DataFrame(data,index=[0])
input_penguins = pd.concat([input_data,x_raw],axis=0)

with st.expander("input features"):
  st.write("**input penguins**")
  input_data
  st.write("**combined penguins data**")
  input_penguins

encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins,prefix=encode)
x_train = df_penguins[1:]
encoded_row = df_penguins[:1]

target_mapper = {'Adelie':0,
                 'Chinstrap':1,
                 'Gentoo':2
                }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander("data preparation"):
  st.write("**encoded input data**")
  encoded_row
  st.write("**encoded output**")
  y
  
  
model = RandomForestClassifier()
model.fit(x_train,y)

pred = model.predict(encoded_row)
pred_prob = model.predict_proba(encoded_row)

df_prediction_prob = pd.DataFrame(pred_prob)
df_prediction_prob.columns = ['Adelie','Chinstrap','Gentoo']
df_prediction_prob.rename(columns={0:'Adelie',
                                   1:'Chinstrap',
                                   2:'Gentoo'})
st.subheader("predicted species")
df_prediction_prob

penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(penguins_species[pred][0]))

