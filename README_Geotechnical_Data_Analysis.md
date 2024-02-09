
# Geotechnical Data Analysis and ML Model Recommendations

This Streamlit application provides an interactive way to upload, analyze, and model geotechnical data with various machine learning algorithms. It is designed to assist geotechnical engineers and data scientists in understanding their data and selecting the best machine learning model for their analysis needs.

## Features

- **Data Upload**: Supports uploading `.csv` and `.xlsx` files for analysis.
- **Data Analysis**: Provides descriptive statistics, correlation matrix visualization, and frequency histograms of the data.
- **Model Recommendations**: Allows users to train a RandomForest or XGBoost model for classification or regression tasks.
- **ANN Optimization**: Utilizes Optuna for hyperparameter tuning of an Artificial Neural Network (ANN) for either classification or regression tasks.

## Installation

To run this application, you need to have Python installed on your system. The application is built using Streamlit, so you will need to install Streamlit along with other dependencies.

```bash
pip install streamlit pandas seaborn matplotlib scikit-learn xgboost optuna tensorflow numpy
```

## Usage

1. Clone the repository or download the application code.
2. Navigate to the application directory in your terminal.
3. Run the application using Streamlit:

```bash
streamlit run your_app_name.py
```

Replace `your_app_name.py` with the path to the Python script if you've named it differently.

## App Modes

- **Data Upload**: Upload your geotechnical data file in `.csv` or `.xlsx` format.
- **Data Analysis**: Explore descriptive statistics, visualize correlation matrices, and generate frequency histograms of your data.
- **Model Recommendations**: Train and evaluate RandomForest or XGBoost models based on your data.
- **ANN Optimization**: Optimize ANN models using Optuna for hyperparameter tuning.

## Dependencies

- Streamlit
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- XGBoost
- Optuna
- TensorFlow
- NumPy

## Contributing

Contributions to improve the application are welcome. Please ensure to follow best practices for code contributions and adhere to the project's code of conduct.

## License

Specify the license under which the application is released, if applicable.

## Contact

For any queries or suggestions, please contact the repository owner.

---

This README provides a basic overview of the application's functionality, installation, and usage instructions. Ensure to update it as needed to reflect any changes or additional features added to the application.
