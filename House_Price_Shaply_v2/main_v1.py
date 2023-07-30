import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import pickle
import os
import gspread
from google.oauth2 import service_account
from datetime import datetime
#import plotly.express as px


pd.set_option('display.float_format', '{:.2f}'.format)
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
    selected_cols = ['LSTAT', 'RM', 'AGE', 'CRIM', 'CHAS']
    
    return X[selected_cols], target

def load_model():
    with open('./House_Price_Shaply_v2/xgboost_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def sheet_url():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    # Load the credentials from the service account JSON file
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes = scope)

    # Authorize the client using the credentials
    gc = gspread.authorize(credentials)

    # Open the Google Sheet by URL
    sheet_url = st.secrets["private_gsheets_url"]
    sheet = gc.open_by_url(sheet_url)

    # Get the first sheet using sheet number
    worksheet = sheet.get_worksheet(0)

    return worksheet

def main():
    st.title("Boston Housing Dataset Price Prediction")

    data_description = """
    - ***LSTAT***: % lower status of the population
    - ***CRIM***: per capita crime rate by town
    - ***CHAS***: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    - ***RM***: average number of rooms per dwelling
    - ***AGE***: proportion of owner-occupied units built prior to 1940
    """

    st.markdown(data_description)



    # Get the current directory
    # current_directory = os.getcwd()
    # Get all items (files and directories) in the current current_directory
    #items = os.listdir(current_directory)
    #[st.write(val) for val in items]

    # Load the data
    X, y = load_data()
    model = load_model()

    col1, col2 = st.columns(2)
    # Display the dataset
    with col1:
        st.subheader("Example Data")
    
        st.write(X.head(8).style\
                        .format("{:.2f}", na_rep=0, subset=['LSTAT', 'AGE', 'RM', 'CRIM'])\
                        .format("{:.0f}", na_rep=0, subset=['CHAS'])\
                        .hide(axis='index'))
    with col2:
        st.subheader("Data Statistics")
        st.write(X.describe().style\
                    .format("{:.1f}", na_rep=0, subset=['LSTAT', 'AGE', 'RM', 'CRIM'])\
                    .format("{:.1f}", na_rep=0, subset=['CHAS'])\
                    .hide(axis='index'))


    model = load_model()

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.initjs()
    # visualize the first prediction's explanation
    #st.components.v1.html(shap.plots.waterfall(shap_values[0]))

    #st.pyplot(shap.plots.force(shap_values[0],matplotlib=True))

    #CRIM = st.number_input("Per capita crime rate (CRIM):", min_value=0.0, value=0.5)
    #ZN = st.number_input("Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN):", min_value=0.0, value=0.5)
    #INDUS = st.number_input("Proportion of non-retail business acres per town (INDUS):", min_value=0.0, value=0.5)
    
    user_input3 = {}
    CHAS = st.sidebar.selectbox("Charles River dummy variable (CHAS):", [0, 1], index=1)
    CRIM = st.sidebar.number_input("Per capita crime rate (CRIM):", min_value=0.0, value=0.5)
    LSTAT = st.sidebar.number_input("% Lower status of the population (LSTAT):", min_value=0.0, value=10.0)
    RM = st.sidebar.number_input("Average number of rooms per dwelling (RM):", min_value=0.0, value=10.0)
    AGE = st.sidebar.number_input("Proportion of owner-occupied units (AGE):", min_value=0.0, value=10.0)

    user_input3 = {
                    'LSTAT':LSTAT,
                    'RM':RM,
                    'AGE':AGE,
                    'CRIM':CRIM,
                    'CHAS':CHAS
                    }


    input_df = pd.DataFrame([user_input3])
    shap_values_top5 = explainer(input_df)
    predict_value = model.predict(input_df)[0]

    #st.write('New Impact')

    st.subheader("Comparing Shap Vlaue and Prediction of Two Examples")

    col1, col2 = st.columns(2)
    # Add content to the first column (col1)
    with col1:
        st.subheader("Randomly Selected")
        rad_exp = X.iloc[0:1,].to_dict(orient='records')[0]
        st.write(rad_exp)
        st.write("Predicted Vlaue : ", model.predict(X.iloc[0:1,])[0])
        fig = shap.plots.waterfall(shap_values[0])
        st.pyplot(fig)


    # Add content to the second column (col2)
    with col2:
        st.subheader("User Provided")
        st.write(user_input3)
        st.write("Predicted Vlaue : ", model.predict(input_df)[0])
        fig = shap.plots.waterfall(shap_values_top5[0])
        st.pyplot(fig)
    
    #st.pyplot(shap.plots.force(shap_values_top5[0],matplotlib=True))
    
    

    st.subheader('Feature Value & Shap Values')
    col = st.selectbox("Select a column you want to analyze", X.columns, index=2)
    fig = shap.plots.scatter(shap_values[:,col], color = shap_values)
    st.pyplot(fig)

    #fig = px.scatter(x=X[col],y=y, labels={'x':col, 'y':'Target Value'})
    #trendline='ols',
    #st.plotly_chart(fig)

    worksheet = sheet_url()

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_values = [current_timestamp, LSTAT, RM, AGE, CRIM, CHAS, float(predict_value)]
    worksheet.append_row(user_values)


    


if __name__ == "__main__":
    main()
