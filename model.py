import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
file_path = "C:/project/time_series_data_human_activities.csv" 
data = pd.read_csv(file_path)

# Inspect Data
print("Initial data shape:", data.shape)
print(data.head())

# Remove Duplicates
data = data.drop_duplicates()
print("Shape after removing duplicates:", data.shape)

# Handle Missing Values
if data.isnull().sum().any():
    print("Missing values detected. Handling missing values...")
    data = data.fillna(data.mean(numeric_only=True))  
    data = data.fillna(data.mode().iloc[0])          
else:
    print("No missing values detected.")
 # Plot Distribution of Activity
plt.figure(figsize=(10, 6))
data['activity'].value_counts().plot(kind='bar', color='red')
plt.title("Distribution of Activity")
plt.xlabel("activity")
plt.ylabel("Frequency")
plt.show()

# Encode Categorical Data
label_encoder = LabelEncoder()
data['activity'] = label_encoder.fit_transform(data['activity'])

# Split Dataset
X = data.drop('activity', axis=1)  
y = data['activity']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=38)

# Scaling Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Train Model
model = RandomForestClassifier(random_state=38)
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save 
import joblib

# Save the trained model
joblib.dump(model, "trained_model.pkl")

#  load the model for reuse
loaded_model = joblib.load("trained_model.pkl")