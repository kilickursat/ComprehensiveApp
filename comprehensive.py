import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.llms import Ollama
import requests 
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Load and preprocess data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file), None
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl'), None
    except Exception as e:
        return None, str(e)

# Updated Hugging Face Inference API integration with error handling
def generate_text_with_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 50}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return "Failed to generate text due to API error."
    except Exception as e:
        st.error(f"An error occurred while calling the API: {str(e)}")
        return "Failed to generate text due to an exception."

# Setting up the page configuration and title
st.set_page_config(page_title='Geotechnical Data Analysis', layout='wide')
st.title('Geotechnical Data Analysis and ML Model Recommendations')

# Initializing session state for dataframe storage
if 'df' not in st.session_state:
    st.session_state.df = None

# Navigation setup in sidebar
st.sidebar.header('Navigation')
app_mode = st.sidebar.radio(
    'Choose the app mode', 
    ['Data Upload', 'Data Analysis', 'Model Recommendations', 'ANN Optimization', 'Text Generation with OpenHermes']
)

# Data Upload mode
if app_mode == 'Data Upload':
    uploaded_file = st.file_uploader("Upload your CSV or Excel file here.", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        data, error = load_data(uploaded_file)
        if data is not None:
            st.session_state.df = data
            st.success('Data loaded successfully!')
            st.write(data.head())
        else:
            st.error(f"Error loading data: {error}")

elif app_mode == 'Data Analysis' and st.session_state.df is not None:
    df = st.session_state.df
    st.header('Data Analysis')
    analysis_options = st.multiselect('Select the types of analysis to perform:',
                                      ['Data Summary', 'Correlation Matrix', 'Frequency Histograms'],
                                      default=['Data Summary'])
    
    if 'Data Summary' in analysis_options:
        with st.expander("Data Summary"):
            st.write(df.describe())
    
    if 'Correlation Matrix' in analysis_options:
        with st.expander("Correlation Matrix"):
            numeric_df = df.select_dtypes(include=[np.number])
            fig = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig)

    if 'Frequency Histograms' in analysis_options:
        with st.expander("Frequency Histograms"):
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            selected_column = st.selectbox('Select Feature', numeric_columns)
            n_bins = st.slider('Number of Bins', min_value=5, max_value=50, value=10)
            fig = px.histogram(df, x=selected_column, nbins=n_bins)
            st.plotly_chart(fig)

elif app_mode == 'Model Recommendations' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable', df.columns)
    task = st.radio("Task Type", ['Classification', 'Regression'])
    model_type = st.radio("Model Selection", ['Random Forest', 'XGBoost'])
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col].astype(np.float32) if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button('Train Model'):
        with st.spinner('Training model...'):
            if task == 'Regression':
                model = RandomForestRegressor() if model_type == 'Random Forest' else xgb.XGBRegressor()
            else:
                model = RandomForestClassifier() if model_type == 'Random Forest' else xgb.XGBClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            st.success('Model training completed!')

            if task == 'Regression':
                st.write('MSE:', mean_squared_error(y_test, predictions))
                st.write('R^2:', r2_score(y_test, predictions))
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=predictions, ax=ax)
                sns.lineplot(x=y_test, y=y_test, color='red', ax=ax)
                st.pyplot(fig)
            else:
                st.text(classification_report(y_test, predictions))
                cm = confusion_matrix(y_test, predictions)
                fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"))
                st.plotly_chart(fig)

elif app_mode == 'ANN Optimization' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable for optimization', df.columns)
    task = st.radio("ANN Task Type", ['Classification', 'Regression'])
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True).astype(np.float32)
    y = df[target_col].astype(np.float32) if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        model = Sequential([
            Dense(trial.suggest_int('units', 16, 128), activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(trial.suggest_float('dropout', 0.1, 0.5)),
            Dense(1, activation='sigmoid' if task == 'Classification' else 'linear')
        ])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy' if task == 'Classification' else 'mean_squared_error',
                      metrics=['accuracy'] if task == 'Classification' else ['mse'])
        model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 10, 100), verbose=0,
                  validation_split=0.1, batch_size=trial.suggest_int('batch_size', 32, 128))
        loss = model.evaluate(X_test, y_test, verbose=0)[0]
        return loss if not np.isnan(loss) else float('inf')

    if st.button('Start ANN Optimization'):
        with st.spinner('Optimizing ANN...'):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)
            st.success('ANN optimization completed!')
            st.write('Best parameters:', study.best_params)

# Text Generation with Hugging Face's Inference API
elif app_mode == 'Text Generation with OpenHermes':
    st.header("Generate Text with Hugging Face's Model")
    
    # Displaying the text area for user input
    user_prompt = st.text_area("Enter your prompt:", height=150)
    
    # Displaying the button to generate text
    if st.button('Generate Text'):
        if user_prompt:
            with st.spinner('Generating text...'):
                try:
                    generated_text = generate_text_with_huggingface(user_prompt)
                    st.text_area("Generated Text:", generated_text, height=250)
                except Exception as e:
                    # Handle any errors during text generation
                    st.error(f"An error occurred: {e}")
        else:
            # Prompt the user to enter text if they haven't
            st.warning('Please enter a prompt.')

# Connect with Me section
st.markdown('---')  # Adds a horizontal line for visual separation
st.header('Connect with Me 🌐')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kursat-kilic-395b5855/)
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kilickursat)
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    [![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://scholar.google.co.jp/citations?user=sNB5IQsAAAAJ&hl=tr)
    ''', unsafe_allow_html=True)
