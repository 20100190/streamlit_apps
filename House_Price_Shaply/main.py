import streamlit as st
import gspread
from google.oauth2 import service_account
import pandas as pd

# Define the scope for accessing Google Sheets
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

# Get all values from the worksheet as a list of lists
rows = worksheet.get_all_values()

# Create a DataFrame from the data
df = pd.DataFrame(rows[1:], columns=rows[0])

st.write(df.values.tolist())

values = df.values.tolist()

worksheet.append_row([15,16])


# Display the DataFrame using Streamlit
st.write(df)