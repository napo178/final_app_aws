
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import shap
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb


# using linear
# Import Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
st.title('App to predict multiple intelligence')

from PIL import Image
image = Image.open('int.png')
st.image(image, caption='Multiple Intelligence')


df=pd.read_csv('clean_intelligence.csv')
st.dataframe(df)
y=df['intelligence_category'] # define Y
X=df[['a_order','a_value','question_id','text_category','id']] # define X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) # create train and test

fig=px.scatter(df, x='id', y='intelligence_category')
st.plotly_chart(fig, use_container_wtext_categoryth=True)



# features for prediction 
# X = sales_join[['a_value', 'text_categorye', 'text_category','text_category','question_text_category']]
#y = sales_join['items_total']

# Add a heading for input features
st.subheader('Enter  Features for Predictions')

 # Rwquest for input fatures, but replod with some default values
a_order= st.number_input('a_order', 0.0)
st.write(' The a_value is :', a_order)

a_value= st.number_input('a_value', 0.0)
st.write(' The a_order is :', a_value)


question_text_category= st.number_input('question_text_category', 0.0)
st.write(' The question_text_category :', question_text_category)



text_category= st.number_input('text_category', 0.0)
st.write('The text_category is', text_category)

id= st.number_input('id', 0.0)
st.write('The id is ', id)







# Load  the model from disk

if st.button("Predict"):
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)
    predict=model.predict([[a_order,a_value,question_text_category,text_category,id]])
    
    
    st.text(f"""
     The intelligence category is :  {predict[0]} 
    """)   
    

    
    if predict==1:
      st.text('The intelligence category is Musical Intelligence ')
    elif predict==2:
      st.text('The intelligence category is Body/Kinesthetic Intelligence ')
    elif predict==3:
      st.text('The intelligence category is Verbal/Linguistic Intelligence ')
    elif predict==4:
      st.text('The intelligence category is Interpersonal Intelligence ')
    elif predict==5:
      st.text('The intelligence category is Logical Mathematical Intelligence' )
    elif predict==6:
      st.text('The intelligence category is Visual/Spatial Intelligenc ')
    elif predict==7:
      st.text('The intelligence category is Naturalistic Intelligence ')








def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



# plotting XAI


# using xgboost

# Import XGBRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Instantiate the XGBRegressor, xg_reg
xg_reg = XGBRegressor()

# Fit xg_reg to training set
xg_reg.fit(X_train, y_train)

# Predict labels of test set, y_pred
y_pred = xg_reg.predict(X_test)







st.title("Explainable XAI")

st.text('The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')

st.text('If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset ')

 # explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))

# visualize the training set predictions
st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)




