import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load and preprocess data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, engine='openpyxl')

st.set_page_config(page_title='Geotechnical Data Analysis', layout='wide')
st.title('Geotechnical Data Analysis and ML Model Recommendations')

if 'df' not in st.session_state:
    st.session_state.df = None

st.sidebar.header('Navigation')
app_mode = st.sidebar.radio('Choose the app mode', ['Data Upload', 'Data Analysis', 'Model Recommendations', 'ANN Optimization'])

if app_mode == 'Data Upload':
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.success('Data loaded successfully!')
        st.write(st.session_state.df.head())

elif app_mode == 'Data Analysis' and st.session_state.df is not None:
    df = st.session_state.df
    st.header('Data Analysis')
    
    with st.expander("Data Summary"):
        st.write(df.describe())
    
    with st.expander("Correlation Matrix"):
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    with st.expander("Frequency Histograms"):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        col1, col2 = st.columns(2)
        with col1:
            selected_column = st.selectbox('Select Feature', numeric_columns)
        with col2:
            n_bins = st.slider('Number of Bins', min_value=5, max_value=50, value=10)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], bins=n_bins, kde=True, ax=ax)
        ax.set_title(f'Histogram of {selected_column}')
        st.pyplot(fig)


elif app_mode == 'Model Recommendations' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable', df.columns)
    task = st.radio("Task Type", ['Classification', 'Regression'])
    model_type = st.radio("Model Selection", ['Random Forest', 'XGBoost'])

    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col].astype(np.float32) if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button('Train Model'):
        if task == 'Regression':
            model = RandomForestRegressor() if model_type == 'Random Forest' else xgb.XGBRegressor()
        else:
            model = RandomForestClassifier() if model_type == 'Random Forest' else xgb.XGBClassifier()
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        if task == 'Regression':
            st.write('MSE:', mean_squared_error(y_test, predictions))
            st.write('R^2:', r2_score(y_test, predictions))
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=predictions, ax=ax)
            sns.lineplot(x=y_test, y=y_test, color='red', ax=ax)
            st.pyplot(fig)
        else:
            st.text(classification_report(y_test, predictions))

elif app_mode == 'ANN Optimization' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable for optimization', df.columns, key='ann_optimization_target')
    task = st.radio("ANN Task Type", ['Classification', 'Regression'], key='ann_task_type')

    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True).astype(np.float32)
    y = df[target_col].astype(np.float32) if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        model = Sequential([
            Dense(trial.suggest_int('units', 16, 128), activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(trial.suggest_float('dropout', 0.1, 0.5)),
            Dense(1, activation='sigmoid' if task == 'Classification' else 'linear')
        ])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        model.compile(optimizer=optimizer, 
                      loss='binary_crossentropy' if task == 'Classification' else 'mean_squared_error',
                      metrics=['accuracy'] if task == 'Classification' else ['mse'])
        model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 10, 100), verbose=0,
                  validation_split=0.1, batch_size=trial.suggest_int('batch_size', 32, 128))
        loss = model.evaluate(X_test, y_test, verbose=0)[0]
        return loss

    if st.button('Start ANN Optimization'):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        st.write('Best parameters:', study.best_params)
