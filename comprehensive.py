import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

# Load data function
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, engine='openpyxl')

# Initialize Streamlit app
st.set_page_config(page_title='Geotechnical Data Analysis', layout='wide')
st.title('Geotechnical Data Analysis and ML Model Recommendations')

if 'df' not in st.session_state:
    st.session_state.df = None

st.sidebar.header('Navigation')
app_mode = st.sidebar.radio('Choose the app mode', ['Data Upload', 'Data Analysis', 'Model Recommendations', 'ANN Optimization'])

# Data Upload
if app_mode == 'Data Upload':
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.success('Data loaded successfully!')
        st.write(st.session_state.df.head())

# Data Analysis
elif app_mode == 'Data Analysis' and st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("Data Summary"):
        st.write(df.describe())
    with st.expander("Data Distribution"):
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

# Model Recommendations
elif app_mode == 'Model Recommendations' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable', df.columns)
    task = st.radio("Task Type", ['Classification', 'Regression'], key='model_task')
    model_type = st.radio("Model Selection", ['Random Forest', 'XGBoost'], key='model_type')
    
    # Preprocessing
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col] if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    if st.button('Train Model'):
        model = None
        if task == 'Regression':
            model = RandomForestRegressor() if model_type == 'Random Forest' else xgb.XGBRegressor()
        elif task == 'Classification':
            model = RandomForestClassifier() if model_type == 'Random Forest' else xgb.XGBClassifier()
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Results display
        if task == 'Regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.write(f'MSE: {mse}, R^2: {r2}')
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=predictions, ax=ax)
            sns.lineplot(y=y_test, x=y_test, color='red', ax=ax)  # Line for perfect predictions
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            st.pyplot(fig)
        else:
            report = classification_report(y_test, predictions, output_dict=False)
            st.text(report)

# ANN Optimization with Optuna
elif app_mode == 'ANN Optimization' and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.selectbox('Select the target variable for ANN optimization', df.columns, key='ann_opt_target')
    task = st.radio("ANN Task Type", ['Classification', 'Regression'], key='ann_task_type')
    
    # Preprocessing for ANN
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col] if task == 'Regression' else LabelEncoder().fit_transform(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def objective(trial):
        model = Sequential()
        model.add(Dense(units=trial.suggest_int('units', 32, 512, log=True), activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
        model.add(Dense(1, activation='sigmoid' if task == 'Classification' else 'linear'))
        optimizer_options = {'rmsprop': RMSprop(), 'adam': Adam(), 'sgd': SGD()}
        model.compile(loss='binary_crossentropy' if task == 'Classification' else 'mean_squared_error',
                      optimizer=optimizer_options[trial.suggest_categorical('optimizer', ['rmsprop', 'adam', 'sgd'])],
                      metrics=['accuracy' if task == 'Classification' else 'mse'])
        
        model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 10, 100, step=10), verbose=0, batch_size=trial.suggest_int('batch_size', 16, 128, log=True))
        loss, _ = model.evaluate(X_test, y_test, verbose=0)
        return loss

    if st.button('Start ANN Optimization'):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Adjust n_trials based on your needs
        st.write('Best parameters:', study.best_params)
