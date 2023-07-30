import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import pickle
import os


st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
    
    return X, target

def load_model():
    with open('./xgboost_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.title("Boston Housing Dataset Exploration")

    # Load the data
    X, y = load_data()
    model = load_model()

    # Display some basic information about the dataset
    st.write("Number of samples:", X.shape[0])
    st.write("Number of features:", X.shape[1])

    # Display the dataset
    st.subheader("Dataset")
    
    st.dataframe(X.head())


    model = load_model()

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.initjs()
    # visualize the first prediction's explanation
    #st.components.v1.html(shap.plots.waterfall(shap_values[0]))

    st.pyplot(shap.plots.force(shap_values[0],matplotlib=True))


    # Create a text input field
    user_input = st.text_input("Enter your First feature here integet:")

    # Display the input value
    st.write("You entered:", user_input)

    X.iloc[0,-1] = int(user_input)

    shap_values_top5 = explainer(X.iloc[0:5,:])
    st.write('New Impact')
    st.pyplot(shap.plots.force(shap_values_top5[0],matplotlib=True))

    shap.initjs()
    fig = shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)


    #st_shap(shap.plots.waterfall(shap_values[0]))
    st.write('End of Code')

if __name__ == "__main__":
    main()
