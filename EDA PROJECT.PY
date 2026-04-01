# step 1
# import liabraries and load dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# load dataset
df=pd.read_csv("Automobile_data.csv")

# view few records of data
print(df.head())

# step 2
# Initial Data Exploration
# 🔸 1. Shape (rows & columns)
print(df.shape)
# 🔸 2. Column names
print(df.columns)
# 🔸 3. Data types
print(df.dtypes)
# 🔸 4. Info summary
df.info()
# 🔸 5. Check missing values
print(df.isnull().sum())
# 🔸 6. Last 40 records
print(df.tail(40))

# 🔹 STEP 3: Data Cleaning
# 🔸 1. Replace '?' with NaN
df.replace("?", np.nan, inplace=True)
# 🔸 2. Convert columns to numeric
df["price"] = pd.to_numeric(df["price"])
df["horsepower"] = pd.to_numeric(df["horsepower"])
# 🔸 3. Handle missing values
# Fill numeric columns with mean
df["price"].fillna(df["price"].mean(), inplace=True)
df["horsepower"].fillna(df["horsepower"].mean(), inplace=True)

# 🔸 4. Outlier Detection (Boxplot)
sns.boxplot(x=df["price"])
plt.show()
#  5. remove outlier
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

df = df[(df["price"] >= Q1 - 1.5*IQR) & (df["price"] <= Q3 + 1.5*IQR)]

# STEP 4: Categorical vs Numerical
num_cols = df.select_dtypes(include=np.number)
cat_cols = df.select_dtypes(exclude=np.number)

print(num_cols.columns)
print(cat_cols.columns)

# 🔸 Encoding (Label Encoding)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols.columns:
    df[col] = le.fit_transform(df[col].astype(str))
    
# 🔹 STEP 5: Skewness Analysis
print(df.skew())

# 🔸 Fix skewness using log transformation
df["price"] = np.log1p(df["price"])
df["horsepower"] = np.log1p(df["horsepower"])

# 🔹 STEP 6: Feature Engineering
# 🔸 Create new feature: area
df["area"] = df["width"] * df["height"]

# 🔸 Drop old columns
df.drop(["width", "height"], axis=1, inplace=True)

# 🔹 STEP 7: Data Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled.head())

# 🔹 STEP 8: Final Dataset
# Features
X = df_scaled.drop("price", axis=1)

# Target
y = df_scaled["price"]

print(X.head())
print(y.head())

# 🔹 STEP 9: Data Visualization (VERY IMPORTANT)
# 🔸 1. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
# 🔸 2. Price vs Horsepower
sns.scatterplot(x=df["horsepower"], y=df["price"])
plt.show()
# 🔸 3. Countplot (Car brands)
sns.countplot(x=df["make"])
plt.xticks(rotation=90)
plt.show()
# 🔸 4. Distribution Plot
sns.histplot(df["price"], kde=True)
plt.show()

df.to_csv("cleaned_automobile_data.csv", index=False)
# ..............................................................................................................................
#  🎯 CREATE DASHBOARD
# -------------------------------
plt.figure(figsize=(18, 12))

# 🔶 TITLE
plt.suptitle("🚗 AUTOMOBILE DATA ANALYSIS DASHBOARD", fontsize=20, fontweight='bold')

# -------------------------------
# 📊 1. Price Distribution
# -------------------------------
plt.subplot(2, 3, 1)
sns.histplot(df["price"], kde=True)
plt.title("Price Distribution")

# -------------------------------
# 📊 2. Horsepower vs Price
# -------------------------------
plt.subplot(2, 3, 2)
sns.scatterplot(x=df["horsepower"], y=df["price"])
plt.title("Horsepower vs Price")

# -------------------------------
# 📊 3. Engine Size vs Price
# -------------------------------
plt.subplot(2, 3, 3)
sns.scatterplot(x=df["engine-size"], y=df["price"])
plt.title("Engine Size vs Price")

# -------------------------------
# 📊 4. Car Brand Count
# -------------------------------
plt.subplot(2, 3, 4)
sns.countplot(x=df["make"])
plt.xticks(rotation=90)
plt.title("Car Brand Distribution")

# -------------------------------
# 📊 5. Fuel Type vs Price
# -------------------------------
plt.subplot(2, 3, 5)
sns.boxplot(x=df["fuel-type"], y=df["price"])
plt.title("Fuel Type vs Price")

# -------------------------------
# 📊 6. Correlation Heatmap
# -------------------------------
plt.subplot(2, 3, 6)
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")

# Layout fix
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show dashboard
plt.show()