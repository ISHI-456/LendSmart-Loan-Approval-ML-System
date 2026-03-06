
import streamlit as st
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LendSmart-CreditWise – Loan Approval System",
    page_icon="💳",
    layout="wide",
)

# ---------------- LOAD MODELS & PREPROCESSORS ----------------
log_model = pickle.load(open("models/logistic_model.pkl", "rb"))
nb_model = pickle.load(open("models/naive_bayes_model.pkl", "rb"))
rf_model = pickle.load(open("models/random_forest_model.pkl", "rb"))
dt_model = pickle.load(open("models/decision_tree_model.pkl", "rb"))
xgb_model = pickle.load(open("models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))
feature_cols = pickle.load(open("models/feature_columns.pkl", "rb"))

def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>💳LendSmart- CreditWise Loan Approval System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning based Loan Approval using Logistic Regression</p>",
    unsafe_allow_html=True,
)

st.subheader("📄 Sample of Training Dataset")
st.write("Showing first 5 rows of the dataset used for training:")
st.dataframe(df.head(), use_container_width=True)





st.divider()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📋 Applicant Details")

Applicant_Income = st.sidebar.number_input("Applicant Income", 0)
Coapplicant_Income = st.sidebar.number_input("Co-applicant Income", 0)
Age = st.sidebar.number_input("Age", 18, 100)
Dependents = st.sidebar.number_input("Dependents", 0, 10)
Existing_Loans = st.sidebar.number_input("Existing Loans", 0, 10)
Savings = st.sidebar.number_input("Savings", 0)
Collateral_Value = st.sidebar.number_input("Collateral Value", 0)
Loan_Amount = st.sidebar.number_input("Loan Amount", 0)
Loan_Term = st.sidebar.number_input("Loan Term (months)", 12, 360)

Education_Level = st.sidebar.selectbox("Education Level", ["Not Graduate", "Graduate"])
Gender_Male = st.sidebar.selectbox("Gender", ["Female", "Male"])

Employment_Status = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Self-employed", "Unemployed","Contract"]
)

Loan_Purpose = st.sidebar.selectbox(
    "Loan Purpose", ["Car", "Education", "Home", "Personal","Business"]
)

Property_Area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

Employer_Category = st.sidebar.selectbox(
    "Employer Category", ["Government", "MNC", "Private", "Unemployed","Business"]
)

Credit_Score = st.sidebar.number_input(
    "Credit Score", min_value=300, max_value=900, value=650
)

# ---------------- FEATURE ENGINEERING (SAME AS TRAINING) ----------------
total_income = Applicant_Income + Coapplicant_Income
dti_ratio = Existing_Loans / total_income if total_income != 0 else 0

le_edu = pickle.load(open("models/le_edu.pkl", "rb"))
edu_encoded = le_edu.transform([Education_Level])[0]

num_df = pd.DataFrame(
    [
        {
            "Applicant_Income": Applicant_Income,
            "Coapplicant_Income": Coapplicant_Income,
            "Age": Age,
            "Dependents": Dependents,
            "Education_Level":edu_encoded,
            "Existing_Loans": Existing_Loans,
            "Savings": Savings,
            "Collateral_Value": Collateral_Value,
            "Loan_Amount": Loan_Amount,
            "Loan_Term": Loan_Term,
            "Total_Income": total_income,
            "DTI_Ratio_sq": dti_ratio**2,
            "Credit_Score_sq": Credit_Score**2,
        }
    ]
)

# ---------------- CATEGORICAL DATA ----------------
cat_df = pd.DataFrame(
    [
        {
            "Employment_Status": Employment_Status,
            "Marital_Status": "Single",  # must match training
            "Loan_Purpose": Loan_Purpose,
            "Property_Area": Property_Area,
            "Employer_Category": Employer_Category,
            "Gender": Gender_Male,
        }
    ]
)

# ---------------- ENCODING ----------------
encoded_cat = encoder.transform(cat_df)
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())

# ---------------- FINAL INPUT ----------------
input_df = pd.concat([num_df, encoded_cat_df], axis=1)
input_df = input_df.reindex(columns=feature_cols, fill_value=0)


# ---------------- PREDICTION ----------------
col1, col2 = st.columns(2)

if st.sidebar.button("🔍 Predict Loan Approval"):
    input_scaled = scaler.transform(input_df)
    xgb_model = xgb_model.predict(input_scaled)[0]
    
    # Display results in columns
    res_col1,res_col2 = st.columns(2)

    with res_col1:
        st.success("XGBoost Prediction:")
        status = "Approved ✅" if xgb_model== 1 else "Rejected ❌"
        st.metric("Status", status)

    

else:
    st.write("👈 Adjust parameters and click 'Predict' to see results.")
st.divider()

# ---------------- MODEL PERFORMANCE ----------------
st.header("📊 Model Performance ")

with open("models/metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

df_metrics = pd.DataFrame(metrics).T
st.dataframe(df_metrics, use_container_width=True)

# ---------------- CHARTS ----------------
st.subheader("📈 Metrics Visualization")



fig, ax = plt.subplots(figsize=(10, 6))
df_metrics.plot(kind="bar", ax=ax, legend=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_ylabel("Score")
ax.set_title("Model Performance Metrics")
plt.tight_layout()  
st.pyplot(fig)


st.subheader("🥧 Loan Approval Distribution")

classes_count =df["Loan_Approved"].value_counts()

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    classes_count,
    labels=["No", "Yes"],
    autopct="%1.1f%%",
    startangle=90
)
ax.set_title("Is loan approved or not?")

st.pyplot(fig)

st.subheader("Employment Category vs Loan Approval Status")

# Group data
employment_approval = (
    df.groupby("Employer_Category")["Loan_Approved"]
      .value_counts()
      .unstack()
      .fillna(0)
)

# Plot
fig, ax = plt.subplots()
employment_approval.plot(kind="bar", ax=ax)

ax.set_xlabel("Employment Category")
ax.set_ylabel("Number of Applications")
ax.set_title("Employment Category vs Loan Approval")

ax.legend(["Rejected", "Approved"])

st.pyplot(fig)


st.subheader("🔥 Feature Correlation Heatmap")

with open("models/corr.pkl", "rb") as f:
    corr = pickle.load(f)

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><center>🚀 Built with Streamlit | CreditWise ML System</center>",
    unsafe_allow_html=True,
)
