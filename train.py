import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv("data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y_encoded)

joblib.dump(model, "sign_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Training done")
