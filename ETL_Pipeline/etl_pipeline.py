import pandas as pd
import sqlite3

# Step 1: Extract - Read data from a CSV file
df = pd.read_csv("employees.csv")

# Step 2: Transform - Clean the data (convert date format, handle missing values)
df['joining_date'] = pd.to_datetime(df['joining_date'])  # Convert to datetime
df.dropna(inplace=True)  # Remove rows with missing values

# Step 3: Load - Store the data in an SQLite database
conn = sqlite3.connect("employees.db")  # Create SQLite database
df.to_sql("employees", conn, if_exists="replace", index=False)

# Verify the data is loaded correctly
query_result = pd.read_sql("SELECT * FROM employees", conn)
print(query_result)

# Close the database connection
conn.close()