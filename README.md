# 📸 London Photographers Analysis & Earnings Prediction

## 📖 Overview
This project analyzes and predicts photographers' earnings in London using data preprocessing, visualization, and machine learning models. It includes:
- **Data Cleaning & Preprocessing** (handling missing values, encoding categorical features)
- **Data Visualization** (histograms, heatmaps, scatter plots, and boxplots)
- **Machine Learning Models** (Linear Regression & Decision Tree Regressor)
- **Model Evaluation** (Mean Squared Error, R-Squared Score)

## 🛠️ Prerequisites
Before running the scripts, ensure you have the following installed:
- **Python 3.x**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

To install the required libraries, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
📂 Project Structure
bash
Copy
Edit
├── london_photographers_dataset.csv   # Input dataset
├── earnings_prediction.py             # Main script for data processing & modeling
├── distribution_of_numerical_features.png  # Histogram output
├── correlation_matrix.png             # Heatmap output
├── scatter_plot_matrix.png            # Pairplot output
├── boxplot_numerical_features.png      # Boxplot output
├── README.md                           # Project documentation
📊 Data Preprocessing
1️⃣ Handling Missing Values
Numerical columns (Earnings_from_Photography, Photos_Taken_Monthly, Social_Media_Following) are filled with their median values.
Categorical columns (Photographer_Name, Lens_Type, Experience_Level, Editing_Software, Weather_Preference) are filled with their mode.
python
Copy
Edit
df["Earnings_from_Photography"] = df["Earnings_from_Photography"].fillna(df["Earnings_from_Photography"].median())
df["Photos_Taken_Monthly"] = df["Photos_Taken_Monthly"].fillna(df["Photos_Taken_Monthly"].median())
df["Social_Media_Following"] = df["Social_Media_Following"].fillna(df["Social_Media_Following"].median())

categorical_cols = ["Photographer_Name", "Lens_Type", "Experience_Level", "Editing_Software", "Weather_Preference"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
2️⃣ Encoding Categorical Features
Label Encoding is applied to categorical features to convert them into numerical values.

python
Copy
Edit
from sklearn.preprocessing import LabelEncoder

categorical_features = ["Camera_Brand", "Camera_Model", "Lens_Type", "Shooting_Style", 
                        "Experience_Level", "Editing_Software", "Location_London", 
                        "Weather_Preference", "Time_of_Shooting"]
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
📊 Data Visualization
1️⃣ Distribution of Numerical Features
A histogram is plotted to show the distribution of numerical features.

python
Copy
Edit
axes = df[numerical_cols].hist(figsize=(12, 8), bins=20, grid=True)
plt.savefig("distribution_of_numerical_features.png", dpi=300)
plt.show()
✅ Output: distribution_of_numerical_features.png

2️⃣ Correlation Heatmap
A heatmap is generated to visualize feature correlations.

python
Copy
Edit
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()
✅ Output: correlation_matrix.png

3️⃣ Scatter Plot Matrix
python
Copy
Edit
sns.pairplot(df[numerical_cols])
plt.savefig("scatter_plot_matrix.png", dpi=300)
plt.show()
✅ Output: scatter_plot_matrix.png

4️⃣ Boxplot for Outlier Detection
python
Copy
Edit
sns.boxplot(data=df[numerical_cols])
plt.savefig("boxplot_numerical_features.png", dpi=300)
plt.show()
✅ Output: boxplot_numerical_features.png

🤖 Machine Learning Models
1️⃣ Data Preparation
Features (X) and target (y) are defined. The dataset is split into training (80%) and test (20%) sets.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Photographer_ID", "Photographer_Name", "Earnings_from_Photography"])
y = df["Earnings_from_Photography"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
2️⃣ Training Models
Linear Regression
python
Copy
Edit
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
Decision Tree Regressor
python
Copy
Edit
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
🏆 Model Evaluation
Two evaluation metrics are used:

R-squared (R²): Measures how well the model explains variance in the target variable.
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
python
Copy
Edit
from sklearn.metrics import mean_squared_error, r2_score

print("Linear Regression:")
print("R-squared:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

print("\nDecision Tree Regression:")
print("R-squared:", r2_score(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))
📢 Results & Conclusion
Linear Regression gives a baseline understanding of how different features influence earnings.
Decision Tree Regressor captures non-linear relationships but may overfit.
Feature engineering and hyperparameter tuning can further improve model performance.
🚀 How to Run the Code
1️⃣ Ensure you have all dependencies installed.
2️⃣ Place london_photographers_dataset.csv in the project directory.
3️⃣ Run the script:

bash
Copy
Edit
python earnings_prediction.py
4️⃣ Review the output in the console and generated visualization files.

🔍 Future Enhancements
Add more advanced models like Random Forest or Gradient Boosting.
Perform feature selection to improve model performance.
Experiment with hyperparameter tuning using GridSearchCV.
Deploy the model using Flask or FastAPI.
📜 License
This project is open-source under the MIT License.