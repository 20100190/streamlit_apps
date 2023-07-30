
import os
import streamlit as st
import plotly.express as px
#import altair as alt
import pandas as pd
import numpy as np


# Configuration
DATA_DIR = "data"
DATA_FILE = "CD.csv"

# Load data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)
@st.cache
def load_data2(nrows, DATA_URL,DATE_COLUMN):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Data processing
def process_data(data):
    # Perform some data processing
    processed_data = data.dropna()
    return processed_data

# Visualization
def visualize_data(data):
    # Visualize the processed data
    st.write(data)

# Main app
def main():
    st.title("Streamlit App")

    # Get the file path for data
    data_path = os.path.join(DATA_DIR, DATA_FILE)

    data_path = './House_Price_Shaply/data/' + DATA_FILE

    # Load and process data
    data = load_data(data_path)
    processed_data = process_data(data)

    # Visualize the processed data
    st.subheader("Processed Data")
    visualize_data(processed_data)

    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

    st.sidebar.header('User Input Parameters')
    # Slider name should be unique 
    # if using st.sidebar then it goes to side bar else in middle
    rate_min, rate_max = st.sidebar.slider('Hourly Rate', 0, 150,[0,150], 5) #st.sidebar.slider to show slider in sadbar
    rate_min_, rate_max_ = st.slider('Hourly Rate_', 0, 150,[0,150], 5)  #(name-of-slider, min-value,max-value, [default min-max], step)
    selected_cluster = st.sidebar.multiselect('Tiers', ['Elite', 'HQ'], ['Elite']) # similar name, options, defualt
    # selected_cluster that will be the value user selected and can be used in query or to filter data


    data_load_state = st.text('Loading data...')
    data = load_data2(100, DATA_URL,DATE_COLUMN)
    data_load_state.text("Done! (using st.cache)")

    #provide a check box for user input
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data.head(rate_min))

    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)


    st.subheader('Number of pickups by hour plotly')
    df = data.groupby(data[DATE_COLUMN].dt.hour).agg({'base':'count'}).reset_index()
    st.table(df.head(3))
    fig = px.bar(df, x="date/time", y='base', barmode='group', title='From wide data', text_auto=True)
    fig.update_layout(showlegend=True,height=400, width=1000, title='Funnel View Different Work Flow Paths')
    fig.update_layout(template="simple_white")
    st.plotly_chart(fig)


    st.table(data.head(3))

    st.table((data.iloc[0:4,:].style
    .background_gradient(cmap = 'BuPu', subset = data.columns[0])
    .background_gradient(cmap = 'Blues', subset = data.columns[1])
    .background_gradient(cmap = 'Reds', subset = data.columns[2])
    .format('{:.3%}', subset = data[['lat']].columns)  #.format('{:.3%}') multiply with 100 and append% and 3 means after decimal show 3 points
    .format('{:.4}', subset = data[['lon']].columns)   #.format('{:.3}') says only show 3 points, including before and after decimal
    ))


    # Some number in the range 0-23
    # provide a slider for user input
    hour_to_filter = st.slider('hour', 0, 23, 10)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.map(filtered_data)



if __name__ == "__main__":
    main()


import streamlit as st
from google.oauth2 import service_account
from gsheetsdb import connect
import pandas as pd
# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)

# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache
def run_query(query):
    rows = conn.execute(query, headers=0)
    return rows

sheet_url = st.secrets["private_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

st.write(pd.DataFrame(rows))
