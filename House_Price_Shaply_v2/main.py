import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt



st.set_option('deprecation.showPyplotGlobalUse', False)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

def main():
    st.title("Boston Housing Dataset Exploration")

    # Load the data
    data, target = load_data()

    # Display some basic information about the dataset
    st.write("Number of samples:", data.shape[0])
    st.write("Number of features:", data.shape[1])

    # Display the dataset
    st.subheader("Dataset")
    X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
    y = target
    st.dataframe(X.head())


    model = xgboost.XGBRegressor().fit(X, y)

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
