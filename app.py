from dotenv import dotenv_values
from openai import OpenAI
import pandas as pd
import streamlit as st

import numpy as np
from io import BytesIO

env = dotenv_values(".env")

st.session_state.setdefault("uploaded_df", None)
st.session_state.setdefault("random_df", None)

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def classify_column(col):
    if pd.api.types.is_numeric_dtype(col):
        if col.nunique() / len(col) > 0.1:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()
if not st.session_state.get("openai_api_key"):
    st.stop()

st.set_page_config(page_title="Wykrywanie najważniejszych danych", layout="centered")
st.title("Wykrywanie najważniejszych danych")

analize_tab, random_data_tab = st.tabs(["Analiza danych", "Generowanie losowego csv"])

with random_data_tab:
    num_rows = st.number_input(
        "Number of rows:", min_value=1, max_value=10000, value=100)
    num_columns = st.number_input(
        "Number of columns:", min_value=1, max_value=100, value=5)
    nan_percentage = st.number_input(
        "Percentage of NaN values:", min_value=0, max_value=100, value=20)
    random_df = None
    if st.button("Generuj losowe dane"):
        column_names = [f"Column_{i+1}" for i in range(num_columns)]
        data = np.random.rand(num_rows, num_columns)
        nan_indices = np.random.choice(
            data.size, size=int(data.size * nan_percentage/100), replace=False)
        data.ravel()[nan_indices] = np.nan
        st.session_state['random_df'] = pd.DataFrame(
            data, columns=column_names)

        st.write("Preview of the generated data:")
        st.dataframe(st.session_state['random_df'].sample(10))

        # Convert DataFrame to CSV
        csv_buffer = BytesIO()
        st.session_state['random_df'].to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Provide download link
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name="random_data.csv",
            mime="text/csv"
        )
    if st.session_state.get('random_df') is not None:
        if st.button("Użyj losowe dane do analizy"):
            st.session_state["uploaded_df"] = st.session_state['random_df']
            print(st.session_state["uploaded_df"].head(3))
    

with analize_tab:
    uploaded_file = st.file_uploader(
        type="csv", help="Załaduj plik CSV z danymi",
        label="Wgraj plik CSV"
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)        
        st.session_state["uploaded_df"] = uploaded_df

    if st.session_state.get("uploaded_df") is not None:
        st.dataframe(st.session_state["uploaded_df"])
        target_column_name = st.selectbox(
            "Wybierz kolumnę docelową:",
            st.session_state["uploaded_df"].columns
        )
        target_column = st.session_state["uploaded_df"][target_column_name]
        
        target_column_type = classify_column(target_column)
        st.write(f"Kolumna docelowa: {target_column_name} jest typu {target_column_type}")





