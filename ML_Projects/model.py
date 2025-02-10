import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("london_photographers_dataset.csv")

# Handling missing values
# Fill numerical missing values with median
df["Earnings_from_Photography"].fillna(df["Earnings_from_Photography"].median(), inplace=True)
df["Photos_Taken_Monthly"].fillna(df["Photos_Taken_Monthly"].median(), inplace=True)
df["Social_Media_Following"].fillna(df["Social_Media_Following"].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = ["Photographer_Name", "Lens_Type", "Experience_Level", "Editing_Software", "Weather_Preference"]
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical features to numerical
label_encoders = {}
categorical_features = ["Camera_Brand", "Camera_Model", "Lens_Type", "Shooting_Style", "Experience_Level", "Editing_Software", "Location_London", "Weather_Preference", "Time_of_Shooting"]

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=["Photographer_ID", "Photographer_Name", "Earnings_from_Photography"])
y = df["Earnings_from_Photography"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate Models
print("Linear Regression:")
print("R-squared:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

print("\nDecision Tree Regression:")
print("R-squared:", r2_score(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))