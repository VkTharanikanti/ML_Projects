import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("london_photographers_dataset.csv")

# Handling missing values
# Fill numerical missing values with median
df["Earnings_from_Photography"] = df["Earnings_from_Photography"].fillna(df["Earnings_from_Photography"].median())
df["Photos_Taken_Monthly"] = df["Photos_Taken_Monthly"].fillna(df["Photos_Taken_Monthly"].median())
df["Social_Media_Following"] = df["Social_Media_Following"].fillna(df["Social_Media_Following"].median())

# Fill categorical missing values with mode for these columns:
categorical_cols = ["Photographer_Name", "Lens_Type", "Experience_Level", "Editing_Software", "Weather_Preference"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert categorical features to numerical
# List of categorical features (as per your data columns)
categorical_features = ["Camera_Brand", "Camera_Model", "Lens_Type", "Shooting_Style", 
                        "Experience_Level", "Editing_Software", "Location_London", 
                        "Weather_Preference", "Time_of_Shooting"]
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

# --------------------
# Visualization Section
# --------------------

# Define numerical columns for visualization
numerical_cols = ["Earnings_from_Photography", "Photos_Taken_Monthly", "Social_Media_Following"]

# 1. Histogram: Distribution of numerical features
axes = df[numerical_cols].hist(figsize=(12, 8), bins=20, grid=True)
for ax in axes.flatten():
    ax.set_ylabel("Frequency")
    ax.set_xlabel(ax.get_xlabel())  # Ensures the x-axis label remains as the column name
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("distribution_of_numerical_features.png", dpi=300)
plt.show()

# 2. Correlation Matrix: Heatmap of correlations among numerical and encoded categorical features
corr = df[numerical_cols + categorical_features].corr()
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

# 3. Scatter Plot Matrix (Pairplot) for selected numerical features
pair_plot = sns.pairplot(df[numerical_cols])
pair_plot.fig.suptitle("Scatter Plot Matrix", fontsize=16, y=1.02)
# Note: Pairplot automatically labels axes with feature names.
plt.savefig("scatter_plot_matrix.png", dpi=300)
plt.show()

# 4. Boxplot: Checking outliers in numerical features
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_cols])
plt.title("Boxplot for Numerical Features", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("boxplot_numerical_features.png", dpi=300)
plt.show()

# --------------------
# Modeling Section
# --------------------

# Preparing Data for Modeling
# Define features (X) and target (y)
# Drop 'Photographer_ID' and 'Photographer_Name' (identifiers) and the target column from X.
X = df.drop(columns=["Photographer_ID", "Photographer_Name", "Earnings_from_Photography"])
y = df["Earnings_from_Photography"]

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Models
print("Linear Regression:")
print("R-squared:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

print("\nDecision Tree Regression:")
print("R-squared:", r2_score(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))