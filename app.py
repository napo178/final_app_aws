
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
# using linear
# Import Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
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




text_category= st.number_input('text_category', 1.0)
st.write('The text_category is', text_category)



id= st.number_input('id', 1.0)
st.write('The id is ', id)







# Load  the model from disk

if st.button("Predict"):
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)
    predict=model.predict([[a_order,a_value,question_text_category,text_category,id]])
    
    
    st.text(f"""
     The intelligence category is :  {predict[0]} 
    """)   
    

y=df['intelligence_category'] # define Y
X=df[['a_order','a_value','question_id','text_category','id']] # define X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) # create train and test   
    
# Initialize LinearRegression model
lin_reg = LinearRegression()

# Fit lin_reg on training data
lin_reg.fit(X_train, y_train)

# Predict X_test using lin_reg
y_pred = lin_reg.predict(X_test)

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Import numpy
import numpy as np

# Compute mean_squared_error as mse
mse = mean_squared_error(y_test, y_pred)

# Compute root mean squared error as rmse
rmse = np.sqrt(mse)

# Display root mean squared error
print("RMSE: %0.2f" % (rmse))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))  

  
shap.initjs() 
explainer = shap.explainers.Linear(lin_reg, X_train)
shap_values = explainer(X_train)

# visualize the model's dependence on the first feature
shap.plots.scatter(shap_values[:, 0])

    
    # Get the input features
    # run predictions





