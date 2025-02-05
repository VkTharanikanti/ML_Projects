import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset

data = sns.load_dataset('iris')

#show first few rows of dataset
print(data.head())

#perform exploratory data analysis (eda):


print(data.describe())
#missingvaluesCheck:
print(data.isnull().sum())

#correlation between numerical values
print(data.corr())

#CREATE VISUALISATIONS:

#(i)Pairplot
sns.pairplot(data, hue = "species")
plt.show()

#(ii) Correlation Heatmap
sns.heatmap(data.corr(), annot = True , cmap = 'coolwarm')
plt.show()

#(iii) Boxplot

sns.boxplot(x = 'species', y = 'sepel_length', data = data)
plt.show()