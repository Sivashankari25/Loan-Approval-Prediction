import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')
data = df.copy()

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check if the 'residential_assets_value' column exists
if 'residential_assets_value' in df.columns:
    df = df[df['residential_assets_value'] >= 10000]
else:
    st.error("Column 'residential_assets_value' does not exist in the dataframe.")
    st.stop()

# Mapping values and stripping whitespace from values before mapping
df['education'] = df['education'].str.strip().map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed'] = df['self_employed'].str.strip().map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].str.strip().map({'Approved': 1, 'Rejected': 0})

# Dropping unnecessary columns
df.drop(['bank_asset_value', 'loan_amount', 'luxury_assets_value'], axis=1, inplace=True)

# Check for and handle NaN values in 'loan_status'
df.dropna(subset=['loan_status'], inplace=True)

# Fill or drop remaining NaN values in features if any
df.fillna(0, inplace=True)

# Ensure there are still rows remaining after filtering and handling NaN values
if df.shape[0] == 0:
    st.error("No data available after filtering and handling NaN values.")
    st.stop()

# Features and target variable
X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

# Scaling numerical features
scaled_columns = ['income_annum', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value']
mms = MinMaxScaler()
X_train[scaled_columns] = mms.fit_transform(X_train[scaled_columns])
X_test[scaled_columns] = mms.transform(X_test[scaled_columns])

# Train the model
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=7)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_forest = rf_clf.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))
print(X.columns)

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon='<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-bank" viewBox="0 0 16 16"><path d="m8 0 6.61 3h.89a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H15v7a.5.5 0 0 1 .485.38l.5 2a.498.498 0 0 1-.485.62H.5a.498.498 0 0 1-.485-.62l.5-2A.5.5 0 0 1 1 13V6H.5a.5.5 0 0 1-.5-.5v-2A.5.5 0 0 1 .5 3h.89zM3.777 3h8.447L8 1zM2 6v7h1V6zm2 0v7h2.5V6zm3.5 0v7h1V6zm2 0v7H12V6zM13 6v7h1V6zm2-1V4H1v1zm-.39 9H1.39l-.25 1h13.72z"/></svg>',
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: green;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title1 {
        color: red;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Loan Approval Prediction')

col1, col2, col3, col4 = st.columns(4)
with col1:
    nof = st.number_input('no_of_dependents')
with col2:
    edu = st.selectbox('education', ['Graduate', 'Not Graduate'])
with col3:
    se = st.selectbox('self_employed', ['Yes', 'No'])
with col4:
    ia = st.number_input('income_annum')

col5, col6, col7, col8 = st.columns(4)
with col5:
    lt = st.number_input('loan_term')
with col6:
    cs = st.number_input('cibil_score')
with col7:
    rav = st.number_input('residential_assets_value')
with col8:
    cav = st.number_input('commercial_assets_value')

col9, col10, col11 = st.columns(3)
with col10:
    btn = st.button('Predict the Approval')

if btn:
    ed_map = {'Graduate': 1, 'Not Graduate': 0}.get(edu, 'Unknown')
    s_map = {'Yes': 1, 'No': 0}.get(se, 'Unknown')

    if ed_map == 'Unknown' or s_map == 'Unknown':
        st.error("Invalid input for education or self_employed.")
    else:
        input_data = pd.DataFrame({
            'no_of_dependents': [nof],
            'education': [ed_map],
            'self_employed': [s_map],
            'income_annum': [ia],
            'loan_term': [lt],
            'cibil_score': [cs],
            'residential_assets_value': [rav],
            'commercial_assets_value': [cav]
        })

        # Scale the input data
        input_data[scaled_columns] = mms.transform(input_data[scaled_columns])

        # Predict and display the result
        try:
            prediction = rf_clf.predict(input_data)
            if prediction == 1:
                st.markdown(f"<h2 class='title'>Yes!! The Loan is Approved</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 class='title1'>No!! The Loan is Not Approved</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
