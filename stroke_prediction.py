import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# missing values
imputer = KNNImputer(n_neighbors=5)
df['bmi'] = imputer.fit_transform(df[['bmi']])

# outlier removal (3.0 IQR — high glucose/bmi বাদ না পড়ে)
for col in ['avg_glucose_level', 'bmi']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 3.0*IQR) & (df[col] <= Q3 + 3.0*IQR)]

# ======================
# ENCODING
# ======================
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

encoders = {}
df_model = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# ======================
# FEATURES
# ======================
X = df_model.drop(['id', 'stroke'], axis=1)
y = df_model['stroke']

feature_columns = X.columns.tolist()

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# balance — SMOTE দিয়ে
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ======================
# MODEL
# ======================
stroke_count = y.value_counts()
weight = stroke_count[0] / stroke_count[1]

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
et = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=weight,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05
)

model = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('xgb', xgb)],
    voting='soft'
)

model.fit(X_train_bal, y_train_bal)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Stroke Prediction AI", page_icon="🧠")
st.title("🧠 Stroke Prediction AI App")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Avg Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)

with col2:
    gender = st.selectbox("Gender", sorted(df['gender'].unique()))
    ever_married = st.selectbox("Ever Married", sorted(df['ever_married'].unique()))
    work_type = st.selectbox("Work Type", sorted(df['work_type'].unique()))
    Residence_type = st.selectbox("Residence Type", sorted(df['Residence_type'].unique()))
    smoking_status = st.selectbox("Smoking Status", sorted(df['smoking_status'].unique()))

# ======================
# PREDICTION
# ======================
if st.button("🔍 Predict", use_container_width=True):

    input_dict = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "smoking_status": smoking_status
    }

    input_data = pd.DataFrame([input_dict])

    # encode
    for col in categorical_cols:
        if input_data[col].iloc[0] in encoders[col].classes_:
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            st.error(f"Invalid value in {col}")
            st.stop()

    # column order fix
    input_data = input_data[feature_columns]

    # scale
    input_scaled = scaler.transform(input_data)

    # predict
    result = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if result[0] == 1:
        st.error(f"⚠️ HIGH STROKE RISK  (Confidence: {proba*100:.1f}%)")
    else:
        st.success(f"✅ LOW RISK  (Stroke probability: {proba*100:.1f}%)")

    # probability bar
    st.markdown("**Stroke Probability**")
    st.progress(float(proba))
    st.caption(f"{proba*100:.1f}% chance of stroke")