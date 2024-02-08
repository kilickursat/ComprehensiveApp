import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, engine='openpyxl')

# Set up Streamlit app
st.set_page_config(page_title='Geotechnical Data Analysis', layout='wide')
st.title('Geotechnical Data Analysis and ML Model Recommendations')

if 'df' not in st.session_state:
    st.session_state.df = None

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
        target_col = st.selectbox('Select the target variable', df.columns)
        df_encoded = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        X = df_encoded
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        task = st.radio("Task Type", ('Classification', 'Regression'))

        if st.button('Train Model'):
            if task == 'Regression':
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                st.write('MSE:', mean_squared_error(y_test, predictions))
                st.write('R2 Score:', r2_score(y_test, predictions))
            else:  # Classification
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                st.write(classification_report(y_test, predictions))

# ANN Optimization with Optuna Section
elif app_mode == 'ANN Optimization':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header('Step 4: Optimize ANN Model')
        target_col = st.selectbox('Select the target variable for optimization', df.columns, key='opt_target')
        df_encoded = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        X = df_encoded
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
            model = Sequential()
            model.add(Dense(units=trial.suggest_int('units_first', 5, 50), activation='relu', input_dim=X_train.shape[1]))
            for i in range(n_layers):
                model.add(Dense(units=trial.suggest_int(f'units_layer_{i}', 5, 50), activation='relu'))
            model.add(Dense(1, activation='sigmoid' if task == 'Classification' else 'linear'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy' if task == 'Classification' else 'mean_squared_error',
                          metrics=['accuracy' if task == 'Classification' else 'mse'])
            model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 5, 100), verbose=0)
            loss, _ = model.evaluate(X_test, y_test, verbose=0)
            return loss

        if st.button('Start Optimization'):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)  # You can adjust the number of trials
            st.write('Best parameters:', study.best_params)
