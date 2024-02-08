import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to load and preprocess data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, engine='openpyxl')

# Initialize Streamlit app and session state for data
st.set_page_config(page_title='Geotechnical Data Analysis', layout='wide')
st.title('Geotechnical Data Analysis and ML Model Recommendations')

if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar for app navigation
st.sidebar.header('Navigation')
app_mode = st.sidebar.radio('Choose the app mode', ['Data Upload', 'Data Analysis', 'Model Recommendations', 'ANN Optimization'])

# Data Upload Section
if app_mode == 'Data Upload':
    st.header('Step 1: Upload Your Dataset')
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.success('Data loaded successfully!')
        st.write(st.session_state.df.head())
    else:
        st.info('Awaiting dataset upload.')

# Data Analysis & Visualization Section
elif app_mode == 'Data Analysis':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header('Step 2: Explore Your Data')
        with st.expander("View Data Summary"):
            st.write(df.describe())

        with st.expander("Visualize Data Distribution"):
            column_to_plot = st.selectbox('Select a column to visualize', df.columns, key='dist_plot')
            fig, ax = plt.subplots()
            sns.histplot(df[column_to_plot], kde=True, ax=ax)
            st.pyplot(fig)

        with st.expander("Correlation Matrix"):
            numeric_df = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    else:
        st.warning('Please upload a dataset in the "Data Upload" section.')

# Model Recommendations Section
elif app_mode == 'Model Recommendations':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header('Step 3: Get Model Recommendations')
        target = st.selectbox('Select the target variable', df.columns)
        
        # Prepare data by encoding categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        task = st.radio("Task Type", ('Classification', 'Regression'))
        
        if st.button('Train Model'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if task == 'Regression':
                model = RandomForestRegressor() if st.radio("Model Selection", ['Random Forest', 'XGBoost'], key='reg_model') == 'Random Forest' else xgb.XGBRegressor()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                st.write(f"MSE: {mse:.2f}, R-squared: {r2:.2f}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=predictions)
                sns.lineplot(x=y_test, y=y_test, color='red')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                st.pyplot(fig)

            elif task == 'Classification':
                model = RandomForestClassifier() if st.radio("Model Selection", ['Random Forest', 'XGBoost'], key='class_model') == 'Random Forest' else xgb.XGBClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                report = classification_report(y_test, predictions)
                
                st.text(report)
    else:
        st.warning('Please upload a dataset in the "Data Upload" section.')

# ANN Optimization with Optuna Section (Placeholder)
elif app_mode == 'ANN Optimization':
    st.header('Step 4: Optimize ANN Model')
    st.markdown("This section will be used for ANN optimization with Optuna. Implementation details to be added.")
    # Placeholder for actual Optuna optimization implementation
