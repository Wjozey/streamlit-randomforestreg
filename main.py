import pandas as pd 
import numpy as np
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

#columns: 
# Weight: in grams
# Lenght1, Lenght2, Lenght3: Different proportions in cm.
# Height: in cm
# Width: in cm
# Species: The only categorical column.


header = st.container()
dataset = st.container()
modelTraining = st.container()



@st.cache()
def load_rename(dataset):
    dataset = pd.read_csv(dataset)
    dataset.rename(columns= {'Length1':'VerLen', 'Length2':'DiagLen', 'Length3':'CrossLen'}, inplace=True)
    dataset = dataset[['Species','VerLen','DiagLen','CrossLen','Height','Width','Weight']]
    return dataset
    



with header:
    st.write("""
    # Fish Weight Prediction App

    This app predicts the **Weight of fish**!
    """)
    st.write('---')

with dataset:
     st.header('Fish Weight Estimation From Measurements')
     st.write('''The aim of this study is to estimate weight of the fish indivuduals from their measurements through using Random Forest regression model. 
     The dataset from [Kaggle]("https://www.kaggle.com/aungpyaeap/fish-market")''')
     data = load_rename('Fish.csv')
     
     
     st.subheader('Species distribution on the dataset')
     chart_data = data['Species'].value_counts()
     st.bar_chart(chart_data)
     
     st.subheader('Preview of dataset')
     st.write(data.head())

with modelTraining:
    st.header('Train the model!')
    st.text('Select the hyperparameters of the model')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=1, max_value=10, value=5, step=1)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limits'])

    sel_col.text('List of input features')
    
    feature_cols = data.columns
    input_feature = sel_col.selectbox("Input feature", feature_cols)

    if n_estimators == 'No limits':
        regr = RandomForestRegressor(max_depth=max_depth)

    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


    Dummydata = pd.get_dummies(data, columns = ['Species'], drop_first=True)
    X = Dummydata.drop(['Weight'], axis=1)
    y = Dummydata['Weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
    RandomForestRegressor( random_state = 0)
    regr.fit(X_train, y_train)
    
    # Prediction
    y_pred = regr.predict(X_test)


    # Evaluation Metrics
    disp_col.subheader('Mean Absolute Error of the model is:')
    disp_col.write(mean_absolute_error(y_test, y_pred))

    disp_col.subheader('Mean Squared Error of the model is:')
    disp_col.write(mean_squared_error(y_test, y_pred))
    
    disp_col.subheader('Root Mean Squared Error of the model is:')
    disp_col.write(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    disp_col.subheader('r2_score of the model is:')
    disp_col.write(r2_score(y_test, y_pred))
    

