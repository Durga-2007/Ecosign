import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def normalize_landmarks(landmarks_array):
    normalized_list = []
    for row in landmarks_array:
        wrist_x, wrist_y, wrist_z = row[0], row[1], row[2]
        norm_row = []
        for i in range(0, len(row), 3):
            norm_row.extend([row[i] - wrist_x, row[i+1] - wrist_y, row[i+2] - wrist_z])
        max_val = max(map(abs, norm_row))
        if max_val > 0:
            norm_row = [val / max_val for val in norm_row]
        normalized_list.append(norm_row)
    return np.array(normalized_list)

data = pd.read_csv("data.csv", header=None)
# Ensure x coords are numeric
# We'll drop rows where the first column has strings (if any headers were saved accidentally)
data = data[pd.to_numeric(data[0], errors='coerce').notnull()]

X = data.iloc[:, :-1].astype(float).values
y = data.iloc[:, -1].values

print("Normalizing training data...")
X_normalized = normalize_landmarks(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Training model...")
model = RandomForestClassifier(n_estimators=200)
model.fit(X_normalized, y_encoded)

joblib.dump(model, "sign_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Training done")
