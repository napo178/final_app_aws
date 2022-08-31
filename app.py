
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('App to predict multiple intelligence')

st.header('data_look')
df=pd.read_csv('clean_intelligence.csv')


st.dataframe(df)


from PIL import Image
image = Image.open('int.png')
st.image(image, caption='Multiple Intelligence')

# features for prediction 
# X = sales_join[['a_value', 'text_categorye', 'id','id','question_id']]
#y = sales_join['items_total']

# Add a heading for input features
st.subheader('Enter  Features for Predictions')

 # Rwquest for input fatures, but replod with some default values
a_order= st.number_input('a_order', 1.0)
st.write(' The a_value is :', a_order)



a_value= st.number_input('a_value', 1.0)
st.write(' The a_value is :', a_value)


question_id= st.number_input('question_id', 1.0)
st.write(' The question_id :', question_id)



text_category= st.number_input('text_category', 1.0)

st.write(' The text_categorye is :', text_category)


id= st.number_input('id', 1.0)
st.write('The id is', id)







# Load  the model from disk

if st.button("Predict"):
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)
    predict=model.predict([[a_order,a_value,question_id,text_category,id]])
  

    st.text(f"""
     The intelligence category is :  {predict[0]} 
    """)    # Get the input features
    # run predictions




