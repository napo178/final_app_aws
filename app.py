
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
st.title('App to predict multiple intelligence')
from PIL import Image
image = Image.open('int.png')
st.image(image, caption='Multiple Intelligence')


st.header('data_look')
df=pd.read_csv('clean_intelligence.csv')
st.dataframe(df)

fig=px.scatter(df, x='text_category', y='intelligence_category')
st.plotly_chart(fig, use_container_wtext_categoryth=True)



# features for prediction 
# X = sales_join[['a_value', 'text_categorye', 'text_category','text_category','question_text_category']]
#y = sales_join['items_total']

# Add a heading for input features
st.subheader('Enter  Features for Predictions')

 # Rwquest for input fatures, but replod with some default values
a_order= st.number_input('a_order', 1.0)
st.write(' The a_value is :', a_order)



a_value= st.number_input('a_value', 1.0)
st.write(' The a_value is :', a_value)


question_text_category= st.number_input('question_text_category', 1.0)
st.write(' The question_text_category :', question_text_category)


options = st.multiselect(
     'Text Category',
     ['Mostly Disagree', 'Slightly Disagree', 'Slightly Agree', 'Mostly Agree'],
     ['Yellow', 'Red'])


st.write('You select :', options)


text_category= st.number_input('text_category', 1.0)
st.write('The text_category is', text_category)


text_category= st.number_input('text_category', 1.0)
st.write('The text_category is', text_category)







# Load  the model from disk

if st.button("Predict"):
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)
    predict=model.predict([[a_order,a_value,question_text_category,text_category,text_category]])
    
    
    st.text(f"""
     The intelligence category is :  {predict[0]} 
    """)   
    
    
    options = st.multiselect(
     'TText Category',
     ['Mostly Disagree', 'Slightly Disagree', 'Slightly Agree', 'Mostly Agree'],
     ['Yellow', 'Red'])


st.write('You select :', options)


    
    # Get the input features
    # run predictions





