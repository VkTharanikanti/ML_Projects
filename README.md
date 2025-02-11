# 📸 London Photographers Analysis & Earnings Prediction

## 📖 Overview
This project analyzes and predicts photographers' earnings in London using data preprocessing, visualization, and machine learning models. It includes:
- **Data Cleaning & Preprocessing** (handling missing values, encoding categorical features)
- **Data Visualization** (histograms, heatmaps, scatter plots, and boxplots)
- **Machine Learning Models** (Linear Regression & Decision Tree Regressor)
- **Model Evaluation** (Mean Squared Error, R-Squared Score)

---

## 🛠️ Prerequisites
Before running the scripts, ensure you have the following installed:
- **Python 3.x**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
📊 Data Preprocessing
1️⃣ Handling Missing Values
Numerical columns (Earnings_from_Photography, Photos_Taken_Monthly, Social_Media_Following) are filled with their median values.
Categorical columns (Photographer_Name, Lens_Type, Experience_Level, Editing_Software, Weather_Preference) are filled with their mode.
2️⃣ Encoding Categorical Features
Label Encoding is applied to categorical features to convert them into numerical values.
📊 Data Visualization
1️⃣ Distribution of Numerical Features
A histogram is plotted to show the distribution of numerical features.
Output: distribution_of_numerical_features.png
2️⃣ Correlation Heatmap
A heatmap is generated to visualize feature correlations.
3️⃣ Scatter Plot Matrix
A pairplot is used to analyze relationships between numerical features.
4️⃣ Boxplot for Outlier Detection
Boxplots help visualize outliers in numerical columns.
🤖 Machine Learning Models
1️⃣ Data Preparation
Features (X) and target (y) are defined.
The dataset is split into training (80%) and test (20%) sets.
2️⃣ Training Models
Linear Regression: A simple model to estimate earnings based on independent variables.
Decision Tree Regressor: Captures non-linear relationships but may overfit.
🏆 Model Evaluation
Two evaluation metrics are used:
R-squared (R²): Measures how well the model explains variance in the target variable.
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
📢 Results & Conclusion
Linear Regression provides a baseline understanding of how different features influence earnings.
Decision Tree Regressor captures complex relationships but might overfit.
Further improvements can be achieved using feature engineering and hyperparameter tuning.
🚀 How to Run the Code
1️⃣ Ensure you have all dependencies installed.
2️⃣ Place london_photographers_dataset.csv in the project directory.
3️⃣ Run the script:
4️⃣ Review the output in the console and generated visualization files.
🔍 Future Enhancements
Add more advanced models like Random Forest or Gradient Boosting.
Perform feature selection to improve model performance.
Experiment with hyperparameter tuning using GridSearchCV.
Deploy the model using Flask or FastAPI.
