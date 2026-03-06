import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


loan_data=pd.read_csv("loan_approval_data.csv")
categorical_col=loan_data.select_dtypes(include=["object"]).columns
numerical_col=loan_data.select_dtypes(include=["number"]).columns


#to handle misdding values we use a imputer
from sklearn.impute import SimpleImputer
num_imp=SimpleImputer(strategy="mean")
loan_data[numerical_col]=num_imp.fit_transform(loan_data[numerical_col])

cat_imp=SimpleImputer(strategy="most_frequent")
loan_data[categorical_col]=cat_imp.fit_transform(loan_data[categorical_col])

loan_data=loan_data.drop("Applicant_ID",axis=1)
#Encoding: we can make use of LabelEncoder for binary encoding and OneHotEncoder for categorical encoding
#LabelEncoder will converts eg: yes: 1, no:0
#one hot encoder will form dffrent columns of the various categorries of a column

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_target = LabelEncoder()
loan_data["Loan_Approved"] = le_target.fit_transform(loan_data["Loan_Approved"])

le_edu = LabelEncoder()
loan_data["Education_Level"] = le_edu.fit_transform(loan_data["Education_Level"])


col=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Employer_Category","Gender"]

one=OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)
encoded=one.fit_transform(loan_data[col])


encoded_df=pd.DataFrame(encoded,columns=one.get_feature_names_out(col),index=loan_data.index)
loan_data=pd.concat([loan_data.drop(columns=col),encoded_df],axis=1)


#Feature engineering



loan_data["DTI_Ratio_sq"] = loan_data["DTI_Ratio"] ** 2
loan_data["Credit_Score_sq"] = loan_data["Credit_Score"] ** 2



X=loan_data.drop(columns=["DTI_Ratio","Credit_Score","Loan_Approved"])
y=loan_data["Loan_Approved"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,random_state=42,test_size=0.2
)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)



#Logistic regression
from sklearn.linear_model import LogisticRegression
Lg_model = LogisticRegression()

Lg_model.fit(X_train_scaled,y_train)

#naive bayes
from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(X_train_scaled,y_train)
y_pred1=gb.predict(X_test_scaled)

#random forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

#Decision tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)


#XGBoost

from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

#evaluation (generally for loan approval we give more preference to precsion then to recall score)

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.metrics import f1_score







metrics = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, Lg_model.predict(X_test_scaled))*100,
        "Precision": precision_score(y_test, Lg_model.predict(X_test_scaled))*100,
        "Recall": recall_score(y_test, Lg_model.predict(X_test_scaled))*100,
        "F1 Score": f1_score(y_test, Lg_model.predict(X_test_scaled))*100,
    },
    "Naive Bayes": {
        "Accuracy": accuracy_score(y_test, y_pred1)*100,
        "Precision": precision_score(y_test, y_pred1)*100,
        "Recall": recall_score(y_test, y_pred1)*100,
        "F1 Score": f1_score(y_test, y_pred1)*100,
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf)*100,
        "Precision": precision_score(y_test, y_pred_rf)*100,
        "Recall": recall_score(y_test, y_pred_rf)*100,
        "F1 Score": f1_score(y_test, y_pred_rf)*100,
    },
    "Decision Tree": {
        "Accuracy": accuracy_score(y_test, y_pred_dt)*100,
        "Precision": precision_score(y_test, y_pred_dt)*100,
        "Recall": recall_score(y_test, y_pred_dt)*100,
        "F1 Score": f1_score(y_test, y_pred_dt)*100,
    },
    "XGBoost": {
        "Accuracy": accuracy_score(y_test, y_pred_xgb)*100,
        "Precision": precision_score(y_test, y_pred_xgb)*100,
        "Recall": recall_score(y_test, y_pred_xgb)*100,
        "F1 Score": f1_score(y_test, y_pred_xgb)*100,
    }

}


num_cols=loan_data.select_dtypes(include="number")
corr=num_cols.corr()

plt.figure(figsize=(12,8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
loan_data["Total_Income"] = (
    loan_data["Applicant_Income"] + loan_data["Coapplicant_Income"]
)

loan_data["DTI_Ratio"] = (
    loan_data["Existing_Loans"] / loan_data["Total_Income"]
).fillna(0)

# ---------------- SAVE MODELS ----------------
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(Lg_model, f)

with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(gb, f)

with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("models/decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/encoder.pkl", "wb") as f:
    pickle.dump(one, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)


with open("models/le_edu.pkl", "wb") as f:

    pickle.dump(le_edu, f)

with open("models/corr.pkl", "wb") as f:
    pickle.dump(corr, f)

with open("models/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)


print("✅ Models trained and saved successfully!")
