import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

# ১. ডেটা লোড করা [৩৫৮]
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# ২. ডেটা ক্লিনিং ও প্রি-প্রসেসিং [৩৬১]
# BMI কলামে মিসিং ভ্যালু পূরণ (KNN Imputer ব্যবহার করে)
imputer = KNNImputer(n_neighbors=5)
df['bmi'] = imputer.fit_transform(df[['bmi']])

# আউটলায়ার রিমুভাল (IQR পদ্ধতি)
for col in ['avg_glucose_level', 'bmi']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]]

# ক্যাটাগরিকাল ডেটা এনকোডিং (Label Encoding) [৩৬১]
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ৩. ফিচার ও টার্গেট আলাদা করা
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# ৪. ডেটা স্প্লিটিং (৮০% ট্রেইনিং, ২০% টেস্টিং) [২৯৫]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ৫. ডেটা ব্যালেন্সিং (Random Over-Sampling - ROS) [২৯৭, ৩৬১]
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# ৬. হাইব্রিড এনসেম্বল মডেল তৈরি (RF + ET + XGB) [২৮২, ৩০১]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# সফট ভোটিং এনসেম্বল
ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('xgb', xgb)],
    voting='soft'
)

# মডেল ট্রেনিং
ensemble_model.fit(X_train_balanced, y_train_balanced)

# ৭. প্রেডিকশন ও মূল্যায়ন [৩০৫, ৩০৬]
y_pred = ensemble_model.predict(X_test)

print("--- মডেল পারফরম্যান্স রিপোর্ট ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ৮. ফিচার ইমপোর্ট্যান্স বা গুরুত্বপূর্ণ লক্ষণগুলো দেখা [৩০২, ৩৬৪]
# (র‍্যান্ডম ফরেস্ট থেকে উদাহরণ হিসেবে নেওয়া হলো)
rf.fit(X_train_balanced, y_train_balanced)
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nসবচেয়ে গুরুত্বপূর্ণ ৩টি লক্ষণ:")
print(importances.sort_values(ascending=False).head(3))