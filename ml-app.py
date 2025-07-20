import streamlit as st
import pandas as pd
from io import StringIO
import chardet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Regression App", layout="wide")

def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
            except:
                pass
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        st.write(f"Encoding categorical columns: {list(cat_cols)}")
        X = pd.get_dummies(X, drop_first=True)

    return X, Y

def build_model(df, model_choice):
    X, Y = preprocess_data(df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100, random_state=parameter_random_state)

    st.markdown('**1.2. Data Splits**')
    st.write('Training set:', X_train.shape)
    st.write('Test set:', X_test.shape)

    st.markdown('**1.3. Variable Details**')
    st.write('Input features:', list(X.columns))
    st.write('Target variable:', Y.name)

    if model_choice == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=parameter_n_estimators,
            random_state=parameter_random_state,
            max_features=parameter_max_features,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score if parameter_bootstrap else False,
            n_jobs=parameter_n_jobs
        )
    elif model_choice == 'Linear Regression':
        model = LinearRegression()
    elif model_choice == 'Decision Tree':
        model = DecisionTreeRegressor(
            random_state=parameter_random_state,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf
        )
    elif model_choice == 'Gradient Boosting':
        model = GradientBoostingRegressor(
            n_estimators=parameter_n_estimators,
            random_state=parameter_random_state,
            max_features=parameter_max_features,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
        )

    model.fit(X_train, Y_train)

    st.subheader('2. Model Performance')
    st.write('**2.1. Training Set**')
    Y_train_pred = model.predict(X_train)
    st.write('R²:', r2_score(Y_train, Y_train_pred))
    st.write('MSE:', mean_squared_error(Y_train, Y_train_pred))

    st.write('**2.2. Test Set**')
    Y_test_pred = model.predict(X_test)
    st.write('R²:', r2_score(Y_test, Y_test_pred))
    st.write('MSE:', mean_squared_error(Y_test, Y_test_pred))

    st.subheader('3. Model Parameters')
    st.json(model.get_params())

    if hasattr(model, 'feature_importances_'):
        st.subheader("4. Feature Importance")
        importance = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importance.sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig)

st.title("🔍 ML Regression App ")
st.markdown("Upload data and train regression models.")

with st.sidebar.header('1. upload your csv file'):
    uploaded_file = st.sidebar.file_uploader("upload csv", type=["csv"])
    st.sidebar.markdown("[example file](read_csv('buisness.csv'))")

with st.sidebar.header('2. Select Parameters'):
    model_choice = st.sidebar.selectbox("Choose Regressor", ['Random Forest', 'Linear Regression', 'Decision Tree', 'Gradient Boosting'])
    split_size = st.sidebar.slider('Train-Test Split Ratio (%)', 10, 90, 80, 5)

if model_choice in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
    with st.sidebar.subheader('Hyperparameters'):
        parameter_n_estimators = st.sidebar.slider('n_estimators', 50, 500, 100, 50)
        parameter_max_features = st.sidebar.selectbox('max_features', ['sqrt', 'log2'])
        parameter_min_samples_split = st.sidebar.slider('min_samples_split', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10, 1, 1)
        parameter_criterion = st.sidebar.selectbox('criterion', ['squared_error', 'absolute_error'])
        parameter_bootstrap = st.sidebar.selectbox('bootstrap (only RF)', [True, False])
        parameter_oob_score = st.sidebar.selectbox('oob_score (only RF)', [False, True])
        parameter_n_jobs = st.sidebar.selectbox('n_jobs', [1, -1])
else:
    parameter_n_estimators = 100
    parameter_max_features = 'sqrt'
    parameter_min_samples_split = 2
    parameter_min_samples_leaf = 1
    parameter_criterion = 'squared_error'
    parameter_bootstrap = True
    parameter_oob_score = False
    parameter_n_jobs = -1

parameter_random_state = 42

st.subheader("1. Dataset")

if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        result = chardet.detect(raw_bytes[:10000])
        encoding = result['encoding'] or 'utf-8'

        decoded = raw_bytes.decode(encoding)
        df = pd.read_csv(StringIO(decoded))

        st.write("**1.1. Preview of dataset**")
        st.dataframe(df.head())

        build_model(df, model_choice)

    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
else:
    st.info('Upload your CSV file or use example data below.')
    if st.button('Use Example Dataset'):
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        Y = pd.Series(data.target, name='Target')
        df = pd.concat([X, Y], axis=1)
        st.write(df.head())
        build_model(df, model_choice)
